from argparse import ArgumentParser
import csv
import re
import json
import random
random.seed(44)
import pandas as pd
from tqdm import tqdm

import icd10
from icdmappings import Mapper

parser = ArgumentParser()
parser.add_argument("--mimic_diagnoses_icd", type=str)
parser.add_argument("--mimic_discharge_notes", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--max_examples", type=int, default=500)
args = parser.parse_args()

mapper = Mapper()
diagnoses_lookup = pd.read_csv(args.mimic_diagnoses_icd)

def get_icd_codes_and_description(hadm_id):
    results = []
    hadm_entries = diagnoses_lookup[diagnoses_lookup["hadm_id"] == int(hadm_id)]
    for code, version in zip(hadm_entries.icd_code, hadm_entries.icd_version):
        if version == 9:
            code = mapper.map(code, source='icd9', target='icd10')

        icd10_entry = icd10.find(code)
        if icd10_entry:
            results.append({"code": code,
                            "description": icd10_entry.description
                            })
    return results


def read_discharge_summaries(path):
    discharge_summaries = []
    count = 0
    with open(path) as f:
        reader = csv.reader(f)
        print("Loading discharge summaries...")
        for i, line in enumerate(reader):
            if i == 0:
                continue
            discharge_summaries.append({"hadm_id": line[2],
                                        "text": line[7]})
            count += 1
    return discharge_summaries



results = []
discharge_summaries = read_discharge_summaries(args.mimic_discharge_notes)
random.shuffle(discharge_summaries)

discharge_summaries = discharge_summaries[:args.max_examples]


for ds in tqdm(discharge_summaries):
    icd_codes = get_icd_codes_and_description(ds["hadm_id"])

    results.append({
        "hadm_id": ds["hadm_id"],
        "text": ds["text"],
        "icd_codes": icd_codes
    })


with open(args.output_file, "a") as f_w:
    for result in results:
        f_w.write(json.dumps(result, ensure_ascii=False) + "\n")