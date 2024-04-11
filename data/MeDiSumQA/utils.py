import csv
import re


keywords = ["discharge instructions"]

def split_to_paragraphs(text):
    parags = re.split(r'\n\s*\n\s*\n', text)
    return [p.strip() for p in parags]

def read_discharge_summaries(path):
    discharge_summaries = []
    with open(path) as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0:
                continue
            discharge_summaries.append((line[0], line[7]))
    return discharge_summaries

def find_instructions(ds):
    ds_parags = split_to_paragraphs(ds)
    instructions = []
    ds_wo_instructions = []
    for parag in ds_parags:
        if any([parag.lower().startswith(kw) for kw in keywords]):
            instructions.append(parag)
        else:
            ds_wo_instructions.append(parag)

    return instructions, "\n\n".join(ds_wo_instructions)

def postprocess_instructions(instructions):
    if isinstance(instructions, list):
        instructions = "\n".join(instructions)
    instructions = "\n".join(instructions.split("\n")[1:])
    if "Followup Instructions" in instructions:
        # Remove empty follow up instructions
        instructions = instructions.split("Followup Instructions")[0]
    return instructions