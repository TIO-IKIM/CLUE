import argparse
import re
import json
import csv
import random
from pathlib import Path

from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import InferenceClient
from evaluate import load
from bert_score import score as b_score

from utils import compute_UMLS_F1, build_few_shot_examples, build_first_turn, update_results, compute_average_results, build_model_input


sys_prompt = "You are a highly skilled assistant, specifically trained to assist lays in understanding and extracting key information from medical documents. Your primary responsibility will be to interprete discharge letters from hospitals. You will receive such a discharge letter. You should carefully review the contents and accurately answer questions related to this document. Only respond with the correct answer to the question. Answer briefly without mentioning a lot of specific details. If the question is about measurements (e.g., lab values), interpret their meaning in relation to the question, rather than writing down the values. Do not generate anything else."

user_prompt_template = """--------------BEGIN DISCHARGE LETTER--------------
{without_discharge_summary}
--------------END DISCHARGE LETTER--------------
Question: {Question}"""

assistant_response_template = """Answer: {Answer}"""

max_new_tokens = 200

def compute_metrics(model_output, label, rouge):
    predictions = [model_output]
    references = [label]
 
 
    rouge_result = rouge.compute(
        predictions=predictions,
        references=references
    )
    
    BERT_P, BERT_R, BERT_F1 = b_score(
        predictions, references, model_type="emilyalsentzer/Bio_ClinicalBERT")
    BERT_P, BERT_R, BERT_F1 = BERT_P.cpu().item(
    ), BERT_R.cpu().item(), BERT_F1.cpu().item()

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
    argument_parser.add_argument("--model_is_instruct", action='store_true')
    argument_parser.add_argument("--model_has_system", action='store_true')
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
    with open(args.data_path, "r") as data_file:
        data = json.load(data_file)
        
    random.seed(1)
    random.shuffle(data)
    

    # Create few shot examples
    chat = None
    if args.num_few_shot_examples > 0:
        chat = build_few_shot_examples(
            data[:args.num_few_shot_examples], sys_prompt, user_prompt_template, assistant_response_template, args.model_has_system, args.model_is_instruct)
    else:
        chat = build_first_turn(sys_prompt, user_prompt=None, assistant_response=None, has_system=args.model_has_system, is_instruct=args.model_is_instruct)

    results = {}
    for i, entry in enumerate((pbar := tqdm(data[args.num_few_shot_examples:]))):

        model_input = build_model_input(entry, user_prompt_template, args.model_is_instruct, chat, tokenizer)
        model_input += assistant_response_template.format(answer="")

        if i == 0:
            # Print first model input to log format
            with open(log_path / "debug_model_input.txt", "w") as f_w:
                f_w.write(model_input)
     
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
                    ).strip()
        if "phi" in args.model_name_or_path.lower() and " <|end|>" in output:
            output = output.split(" <|end|>")[0]
        # Update metric variables
        new_results = compute_metrics(output, entry["Answer"], rouge)
        update_results(results, new_results)
        average_results = compute_average_results(results)
        
        # Print metrics
        print_metrics = ["ROUGE_L", "BERT_F1", "UMLS_F1"]
        pbar.set_description(
            ", ".join(f"Average {k}: {v:.2f}" for k, v in average_results.items() if k in print_metrics))

        new_results["Question"] = entry["Question"]
        new_results["Model Answer"] = output
        new_results["Ground Truth Answer"] = entry["Answer"]
        with open(log_path / "predictions.json", "a") as out_file:
            json.dump(new_results, out_file)
            out_file.write("\n")

    with open(log_path / "results.json", "w") as f_w:
        json.dump(average_results, f_w)


if __name__ == "__main__":
    main()
