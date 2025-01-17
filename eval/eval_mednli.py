
import argparse
import json
import random
import re
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams

from utils import build_few_shot_examples, build_model_input, update_results, compute_average_results

sys_prompt = """You are a highly skilled assistant, specifically trained to assist medical professionals. You will recieve two sentences, labeled 'SENTENCE_1' and 'SENTENCE_2', respectively. Your task is to determine the logical relation between the two sentences. Valid answers are: ENTAILMENT, NEUTRAL or CONTRADICTION.""" 

user_prompt_template = """
SENTENCE_1: {sentence1}
SENTENCE_2: {sentence2}
"""

assistant_response_template =  """{gold_label}"""

ground_truth_key = "gold_label"
max_tokens = 20


def compute_metrics(model_output, label):
    answer = re.findall(label.upper(), model_output.upper())
    if len(answer) == 1:
        return {"ACCURACY": 1}
    return {"ACCURACY": 0}
    

def main():

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--model", type=str, default="microsoft/phi-4")
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

    # Load data
    with open(args.data_path, "r") as data_file:
        data = [json.loads(line) for line in data_file]
        
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

    for i, (sample, pred) in enumerate(zip(data[args.num_few_shot_examples:], model_predictions)):
        if i == 0:
            # Print first model input to log format
            with open(log_path / "debug_model_input.txt", "w") as f_w:
                f_w.write(pred.prompt)
        pred = pred.outputs[0].text
        ground_truth = sample[ground_truth_key]

        # Update metric variables
        new_results = compute_metrics(pred, ground_truth)
        update_results(results, new_results)
        average_results = compute_average_results(results)
        
        new_results["Model Answer"] = pred
        new_results["Ground Truth Answer"] = ground_truth
        with open(log_path / "predictions.json", "a") as out_file:
            json.dump(new_results, out_file)
            out_file.write("\n")

    with open(log_path / "results.json", "w") as f_w:
        json.dump(average_results, f_w)


if __name__ == "__main__":
    main()