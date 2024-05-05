import argparse
import json
from tqdm import tqdm
import re
from pathlib import Path
from bert_score import score as b_score
from transformers import AutoTokenizer
from huggingface_hub import InferenceClient
from evaluate import load

from utils import build_few_shot_examples, build_first_turn, update_results, compute_average_results, build_model_input, compute_icd_f1, is_icd10_valid, parse_icd_codes


sys_prompt = "You are a highly skilled and detail-oriented assistant, specifically trained to assist medical professionals in interpreting and extracting key information from medical documents. Your primary responsibility will be to analyze discharge letters from hospitals. You will be given such a discharge letter. Your task is to identify all primary and secondary diagnoses from the report and list their respective ICD-10 codes."

user_prompt_template = """--------------BEGIN DISCHARGE LETTER--------------
{text}
--------------END DISCHARGE LETTER--------------
Now return the list of diagnoses ICD-10 codes you found. Only list the ICD-10 codes. Do not generate anything else."""

assistant_response_start = "ICD-10 Codes: "
assistant_response_template =  assistant_response_start + """{codes}"""

ground_truth_key = "codes"


def compute_metrics(predictions, ground_truth):
    ground_truth = ground_truth.split(", ")
    predictions = [pred.replace(".", "") for pred in predictions]

    valid_predictions = [pred for pred in predictions if is_icd10_valid(pred)]
    if len(predictions) > 0:
        valid_pred_ratio = len(valid_predictions) / len(predictions)
    else:
        valid_pred_ratio = 0.0
    P_EM, R_EM, F_EM = compute_icd_f1(predictions, ground_truth, approximate=False)
    P_APPROX, R_APPROX, F_APPROX = compute_icd_f1(predictions, ground_truth, approximate=True)

    return {"ICD EM P": P_EM, "ICD EM R" : R_EM, "ICD EM F1": F_EM,
            "ICD AP P": P_APPROX, "ICD AP R" : R_APPROX, "ICD AP F1": F_APPROX,
            "VALID CODES": valid_pred_ratio}

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

    # Load data
    with open(args.data_path, "r") as data_file:
        data = json.load(data_file)
        
    samples = []
    
    for entry in data:
        samples.append({
            "text": entry["text"],
            "codes": ", ".join([diag['code'] for diag in entry["icd_codes"]])
        })

    chat = None
    if args.num_few_shot_examples > 0:
        chat = build_few_shot_examples(
            samples[:args.num_few_shot_examples], sys_prompt, user_prompt_template, assistant_response_template, args.model_has_system, args.model_is_instruct)
    else:
        chat = build_first_turn(sys_prompt, user_prompt=None, assistant_response=None, has_system=args.model_has_system, is_instruct=args.model_is_instruct)

    results = {}
    for i, entry in enumerate((pbar := tqdm(samples[args.num_few_shot_examples:]))):
        model_input = build_model_input(entry, user_prompt_template, args.model_is_instruct, chat, tokenizer)
        model_input += assistant_response_template.format(**{ground_truth_key: ""})
        
        if i == 0:
            # Print first model input to log format
            with open(log_path / "debug_model_input.txt", "w") as f_w:
                f_w.write(model_input)

        ground_truth = entry[ground_truth_key]

        if "llama-3" in args.model_name_or_path.lower() or "llama3" in args.model_name_or_path.lower():
                    output = inference_client.text_generation(
                    model_input,
                    max_new_tokens=200,
                    stream=False,
                    details=False,
                    stop_sequences=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|im_end|>"],
                    truncate=7992
                    )
                    if "<|im_end|>" in output:
                        output = output.split("<|im_end|>")[0]
        else:
            output = inference_client.text_generation(
                model_input,
                max_new_tokens=200,
                stream=False,
                details=False
            )
        
        # Cut off new self-prompting
        output = re.sub(
            "(You are an AI.*)|(\[INST\].*)|((<\|user\|>).*)", "", output)
        
        output_codes = parse_icd_codes(output)

        # Update metric variables
        new_results = compute_metrics(output_codes, ground_truth)
        update_results(results, new_results)
        average_results = compute_average_results(results)
        
        # Print metrics
        print_metrics = ["ICD EM F1", "ICD AP F1", "VALID CODES"]
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