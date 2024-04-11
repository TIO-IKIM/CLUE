import json
from copy import deepcopy
from tqdm import tqdm
from evaluate import load
import spacy
import scispacy
from argparse import ArgumentParser


from scispacy.linking import EntityLinker
from bert_score import score as b_score
nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
linker = nlp.get_pipe("scispacy_linker")


def get_UMLS_entities(doc):    
    entities = set()
    for entity in doc.ents:
        if entity._.kb_ents:
            entities.add(linker.kb.cui_to_entity[entity._.kb_ents[0][0]].canonical_name)
    return entities

def compute_UMLS_F1(model_output, label):
    doc_model_output = nlp(model_output)
    doc_label = nlp(label)

    model_output_entities = get_UMLS_entities(doc_model_output)
    label_entities = get_UMLS_entities(doc_label)

    if len(model_output_entities) == 0:
        P = 0.0
    else:
        P = len([pred for pred in model_output_entities if pred in label_entities]) / len(model_output_entities)
    if len(label_entities) == 0:
        R = 0.0
    else:
        R = len([l for l in label_entities if l in model_output_entities]) / len(label_entities)

    if (P + R) == 0:
        F = 0.0
    else:
        F = 2 * P * R / (P + R)

    return P, R, F


parser = ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--top_k_qas", type=int)
parser.add_argument("--output_path", type=str)
args = parser.parse_args()

rouge = load("rouge")


with open(args.dataset) as f:
    dataset = [json.loads(l) for l in f]

scored_dataset = []
    
for sample in tqdm(dataset, desc="Scoring examples..."):
    
    for q, a in sample["qas"]:
        qas = {}
        qas["note_id"] = sample["note_id"]
        qas["with_discharge_summary"] = sample["with_discharge_summary"]
        qas["without_discharge_summary"] = sample["without_discharge_summary"]
        qas["rouge_l"] = rouge.compute(predictions=[q], references=[a])["rougeL"]
        qas["bertscore"] = b_score([q], [a], model_type="emilyalsentzer/Bio_ClinicalBERT")[2].item()
        qas["umls_score"] = compute_UMLS_F1(q, a)[2]
        qas["Question"] = q
        qas["Answer"] = a
        scored_dataset.append(qas)
    
    

scored_dataset.sort(key=lambda x: x["rouge_l"])
data_rouge = scored_dataset
data_bertscore = deepcopy(data_rouge)
data_bertscore.sort(key=lambda x: x["bertscore"])

for i in range(args.top_k_qas):
    with open(args.output_path, "a") as f_w:
        if i % 2 == 0:
            sample = data_rouge[i]
        else:
            sample = data_bertscore[i]
        
        f_w.write(json.dumps({
            "note_id": sample["note_id"],
            "with_discharge_summary": sample["with_discharge_summary"],
            "without_discharge_summary": sample["without_discharge_summary"],
            "Question": sample["Question"],
            "Answer": sample["Answer"]
        }) + "\n")