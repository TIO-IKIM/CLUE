
import argparse
import json
import csv
import random
from pathlib import Path
import re

from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import InferenceClient
from evaluate import load
from bert_score import score as b_score

from utils import compute_UMLS_F1, build_few_shot_examples, update_results, compute_average_results, build_model_input

sys_prompt = "You are a highly skilled and detail-oriented assistant, specifically trained to assist medical professionals in interpreting and extracting key information from medical documents. Your primary responsibility will be to analyze discharge letters from hospitals. You will receive an excerpt of such a discharge letter. Your task is to summarize the diagnoses and problems that led to the patient's hospitalization."

user_prompt_template = """--------------BEGIN DISCHARGE LETTER--------------
{Subjectives}

{Assessment}
--------------END DISCHARGE LETTER--------------
Now respond with the list of diagnoses and patient problems. Do not generate anything else."""

assistant_response_start = "Diagnoses/Patient problems: "
assistant_response_template =  assistant_response_start + """{Summary}"""

ground_truth_key = "Summary"
max_new_tokens = 100

def compute_metrics(model_output, label, rouge):
    # capitalization is not uniform in the dataset
    predictions = [model_output.lower()]
    references = [label.lower()]

    rouge_result = rouge.compute(
        predictions=predictions,
        references=references
    )
    BERT_P, BERT_R, BERT_F1 = b_score(predictions, references, model_type="emilyalsentzer/Bio_ClinicalBERT")
    BERT_P, BERT_R, BERT_F1 = BERT_P.cpu().item(), BERT_R.cpu().item(), BERT_F1.cpu().item()

    UMLS_P, UMLS_R, UMLS_F1, = compute_UMLS_F1(model_output, label)
    return {"ROUGE_L": rouge_result["rougeL"],
            "ROUGE1": rouge_result["rouge1"],
            "ROUGE2": rouge_result["rouge2"],
            "BERT_P": BERT_P, "BERT_R": BERT_R, "BERT_F1": BERT_F1, 
            "UMLS_P": UMLS_P, "UMLS_R": UMLS_R, "UMLS_F1": UMLS_F1}
    

def main():

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--model_address", type=str)
    argument_parser.add_argument("--model_name_or_path", type=str)
    argument_parser.add_argument("--model_has_system", action='store_true')
    argument_parser.add_argument("--model_is_instruct", action='store_true')
    argument_parser.add_argument("--num_few_shot_examples", type=int)
    argument_parser.add_argument("--data_path", type=str)
    argument_parser.add_argument("--log_path", type=str)
    argument_parser.add_argument("--token", type=str)

    args = argument_parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        log_path.mkdir(parents=True, exist_ok=True)
    if (log_path / "results.json").exists():
        print(f"Skipping dataset {args.data_path} as results already exist")
        return

    if (log_path / "predictions.json").exists():
        (log_path / "predictions.json").unlink()

    # Tokenizer & Inference client & metrics
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=args.token)
    inference_client = InferenceClient(model=args.model_address)
    model_config = AutoConfig.from_pretrained(args.model_name_or_path, token=args.token, trust_remote_code=True)
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

    # Create few shot examples
    few_shot_chat = build_few_shot_examples(data[:args.num_few_shot_examples], sys_prompt, user_prompt_template, assistant_response_template, args.model_has_system, args.model_is_instruct)

    results = {}

    for i, entry in enumerate((pbar := tqdm(data[args.num_few_shot_examples:]))):
        model_input = build_model_input(entry, user_prompt_template, args.model_is_instruct, few_shot_chat, tokenizer)
        model_input += assistant_response_template.format(**{ground_truth_key: ""})
        
        if i == 0:
            # Print first model input to log format
            with open(log_path / "debug_model_input.txt", "w") as f_w:
                f_w.write(model_input)

        ground_truth = entry[ground_truth_key]

        stop_sequences = None
        if "llama-3" in args.model_name_or_path.lower() or "llama3" in args.model_name_or_path.lower():
            stop_sequences=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|im_end|>"]
        output = inference_client.text_generation(
            model_input,
            max_new_tokens=max_new_tokens,
            truncate=model_config.max_position_embeddings - max_new_tokens,
            stream=False,
            details=False,
            stop_sequences=stop_sequences
            )
        if "phi" in args.model_name_or_path.lower() and " <|end|>" in output:
            output = output.split(" <|end|>")[0]

        # Cut off new self-prompting
        output = re.sub(r"(You are an AI.*)|(\[INST\].*)|((<\|user\|>).*)", "", output)
        
        new_results = compute_metrics(output, entry["Summary"], rouge)
        update_results(results, new_results)
        average_results = compute_average_results(results)
        
        # Print metrics
        print_metrics = ["ROUGE_L", "BERT_F1", "UMLS_F1"]
        pbar.set_description(
            ", ".join(f"Average {k}: {v:.2f}" for k, v in average_results.items() if k in print_metrics))

        new_results["Model Answer"] = output
        new_results["Ground Truth Answer"] = ground_truth
        with open(log_path / "predictions.json", "a") as out_file:
            json.dump(new_results, out_file)
            out_file.write("\n")

    with open(log_path / "results.json", "w") as f_w:
        json.dump(average_results, f_w)


if __name__ == "__main__":
    main()