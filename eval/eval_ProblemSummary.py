import argparse
import re
import json
import csv
import random
from pathlib import Path
import numpy as np

from tqdm import tqdm
import pandas as pd
from evaluate import load
from bert_score import score as b_score
from vllm import LLM, SamplingParams

from utils import compute_UMLS_F1, build_few_shot_examples, build_first_turn, update_results, compute_average_results, build_model_input

sys_prompt = "You are a highly skilled and detail-oriented assistant, specifically trained to assist medical professionals in interpreting and extracting key information from medical documents. Your primary responsibility will be to analyze discharge letters from hospitals. You will receive an excerpt of such a discharge letter. Your task is to summarize the diagnoses and problems that led to the patient's hospitalization."

user_prompt_template = """--------------BEGIN DISCHARGE LETTER--------------
{Subjectives}

{Assessment}
--------------END DISCHARGE LETTER--------------
Now respond with the list of diagnoses and patient problems. Do not generate anything else."""

assistant_response_template =  """{Summary}"""

ground_truth_key = "Summary"
max_tokens = 512

def compute_metrics(predictions, references, rouge):
 
    rouge_result = rouge.compute(
        predictions=predictions,
        references=references,
        use_aggregator=False
    )
    (P, R, F) = b_score(
        predictions, references, lang="en-sci")
    
    UMLS_scores = np.array([compute_UMLS_F1(pred, label) for (pred, label) in zip(predictions, references)])

    return {"ROUGE_L": rouge_result["rougeL"],
            "ROUGE1": rouge_result["rouge1"],
            "ROUGE2": rouge_result["rouge2"],
            "BERT_P": P.cpu().numpy(), "BERT_R": R.cpu().numpy(), "BERT_F1": F.cpu().numpy(),
            "UMLS_P": UMLS_scores[:, 0], "UMLS_R": UMLS_scores[:, 1], "UMLS_F1": UMLS_scores[:, 2]}
    

def main():

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--model", type=str)
    argument_parser.add_argument("--num_few_shot_examples", type=int, default=3)
    argument_parser.add_argument("--data_path", type=str)
    argument_parser.add_argument("--log_path", type=str)
    argument_parser.add_argument("--token", type=str)
    args = argument_parser.parse_args()

    if args.token:
        login(args.token)

    log_path = Path(args.log_path)
    if not log_path.exists():
        log_path.mkdir(parents=True, exist_ok=True)
        
    if (log_path / "results.json").exists():
        print(f"Skipping dataset {args.data_path} as results already exist")
        return


    if (log_path / "predictions.json").exists():
        (log_path / "predictions.json").unlink()

    llm = LLM(args.model, download_dir="./cache")
    sampling_params = SamplingParams(max_tokens=max_tokens)

    rouge = load("rouge")

     # Load data
    with open(args.data_path, "r", newline="") as csvfile:
        csv_reader = csv.reader(csvfile)
        headers = next(csv_reader)
        data = [{header: value for header, value in zip(headers, row)} for row in csv_reader]
        for i in range(len(data)):
            if data[i]["Summary"][-1] == ";":
                data[i]["Summary"][:-1]
            data[i]["Summary"] = data[i]["Summary"].replace(";", ", ")
        
    random.seed(1)
    random.shuffle(data)

    # Create few shot examples
    chat = None
    if args.num_few_shot_examples > 0:
        chat = build_few_shot_examples(
            data[:args.num_few_shot_examples], sys_prompt, user_prompt_template, assistant_response_template)
    else:
        chat = build_first_turn(sys_prompt, user_prompt=None, assistant_response=None)


    results = {}

    model_inputs = [build_model_input(entry, user_prompt_template, chat) for entry in data[args.num_few_shot_examples:]]

    model_predictions = llm.chat(messages=model_inputs, sampling_params=sampling_params)

    # Print first model input to log format
    with open(log_path / "debug_model_input.txt", "w") as f_w:
        f_w.write(model_predictions[0].prompt)

    model_predictions = [pred.outputs[0].text for pred in model_predictions]
    ground_truths = [sample[ground_truth_key] for sample in data[args.num_few_shot_examples:]]

    scores = compute_metrics(model_predictions, ground_truths, rouge)

    for i, (sample, pred) in enumerate(zip(data[args.num_few_shot_examples:], model_predictions)):

        log_dict = {
            "Model Answer": pred,
            "Ground Truth Answer": sample[ground_truth_key],
        }
        for metric in scores:
            log_dict[metric] = "{:.2f}".format(scores[metric][i])
        with open(log_path / "predictions.json", "a") as out_file:
            json.dump(log_dict, out_file)
            out_file.write("\n")

    average_results = {}
    for metric in scores:
        average_results[metric] =  "{:.2f}".format(np.mean(scores[metric]))


    with open(log_path / "results.json", "w") as f_w:
        json.dump(average_results, f_w)


if __name__ == "__main__":
    main()