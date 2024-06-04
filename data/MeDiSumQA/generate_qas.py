import re
import csv
import json
from tqdm import tqdm
import random
random.seed(44)
from argparse import ArgumentParser

from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
from difflib import SequenceMatcher
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")


from prompts import fact_extraction_system_prompt, fact_extraction_user_template, fact_extraction_answer, question_generation_system_prompt, questions_generation_user_template, question_generation_example_input, question_generation_example_output
from utils import split_to_paragraphs, read_discharge_summaries, find_instructions, postprocess_instructions


statement_blocklist = ["the patient is instructed", "the patient is advised","the patient was instructed", "the patient was advised", "the patient should", "the patient needs", "the patient will need", "discharge", "follow-up"]
questions_blocklist = ["follow-up", "discharge", "adviced", "recommended", "next", "care", "should", "dietary"]


def postprocess_llm_results(result):
    lines = result.split("\n")
    processed_lines = []

    for l in lines:
        l = l.strip()
        if "___" in l:
            continue
        # if any([s in l.lower() for s in statement_blocklist]):
        #     break
        if l.startswith("* ") or l.startswith("- "):
            processed_lines.append(l[2:].strip())
    return processed_lines

def postprocess_qas(statements, model_generation):
    statements = [s.strip() for s in statements]
    
    pattern = r"Question: (.*?)\nAnswer: (.*?)\n"
    qas = re.findall(pattern, model_generation)
    qas_filtered = []
    for q, a in qas:
        if a.strip() in statements:
            # if not any(block_w in q.lower() for block_w in questions_blocklist):
            qas_filtered.append((q.strip(), a.strip()))
    return qas_filtered



parser = ArgumentParser()
parser.add_argument("--mimic_discharge_notes", type=str, help="The path to the MIMIC IV discharge notes.")
parser.add_argument("--max_discharge_notes", type=int, default=4000, help="Maximum numbers of discharge notes to process.")
parser.add_argument("--example_ds", type=str)
parser.add_argument("--example_ds_statements", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--model_address", type=str)
args = parser.parse_args()

client = InferenceClient(model=args.model_address)

example_ds = open(args.example_ds).read()
example_ds_statements = open(args.example_ds_statements).read()


count = 0
discharge_summaries = read_discharge_summaries(args.mimic_discharge_notes)
random.shuffle(discharge_summaries)
discharge_summaries = [(id, ds) for id, ds in discharge_summaries if "You were admitted to the hospital" in ds]
for note_id, ds in tqdm(discharge_summaries, total=args.max_discharge_notes):
    instructions, ds_wo_instructions = find_instructions(ds)
    
    instructions = postprocess_instructions(instructions)

    instructions = instructions.split("It was a pleasure")[0]

    if instructions:
        messages = [{"role": "system", "content": fact_extraction_system_prompt},
                    {"role": "user", "content": fact_extraction_user_template.format(instructions=example_ds)},
                    {"role": "assistant", "content": fact_extraction_answer.format(statements=example_ds_statements)}, 
                    {"role": "user", "content": fact_extraction_user_template.format(instructions=instructions)}]
        message_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "\nStatements:\n- "
        result_answers = "\nStatements:\n- " + client.text_generation(prompt=message_formatted, max_new_tokens=1000, stop_sequences=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|im_end|>"])
        statements = postprocess_llm_results(result_answers)
        statements_str = "- " + "\n- ".join(statements)
        
        
        messages = [
            {"role": "system", "content": question_generation_system_prompt},
            {"role": "user", "content": question_generation_example_input}, 
            {"role": "assistant", "content": question_generation_example_output}, 
            {"role": "user", "content": questions_generation_user_template.format(statements=statements_str)}]
        message_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "\nQuestion: "
        qas = "\nQuestion: " + client.text_generation(prompt=message_formatted, max_new_tokens=1000, stop_sequences=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|im_end|>"])
        qas_post = postprocess_qas(statements, qas)
        
        if result_answers:
            count += 1
            with open(args.output_file, "a") as f_a:
                json.dump({
                    "note_id": note_id,
                    "with_summary": ds,
                    "without_summary": ds_wo_instructions,
                    "qas": qas_post,
                }, f_a, ensure_ascii=False)
                f_a.write("\n")
            if count >= args.max_discharge_notes:
                break