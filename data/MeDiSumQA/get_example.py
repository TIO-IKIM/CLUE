from argparse import ArgumentParser
import csv


from utils import find_instructions, postprocess_instructions

parser = ArgumentParser()
parser.add_argument("--mimic_discharge_notes", type=str)
args = parser.parse_args()

with open(args.mimic_discharge_notes) as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0:
                continue
            id = line[0]
            text = line[7]
            if "13714762-DS-14" == id:
                instructions, ds_wo_instructions = find_instructions(text)
                instructions = postprocess_instructions(instructions)
                instructions = instructions.split("It was a pleasure")[0]
                
                with open("data/MeDiSumQA/example_summary.txt", "w") as f_w:
                    f_w.write(instructions)
                    break