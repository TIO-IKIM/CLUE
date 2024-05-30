from statistics import mean
import re

import icd10

nlp = None
Linker = None

def build_first_turn(sys_prompt, user_prompt, assistant_response, has_system, is_instruct):
    if is_instruct:
        if has_system:
            chat = [{"role": "system", "content": sys_prompt}]
            if user_prompt and assistant_response:
                chat.extend([{"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response}])
        else:
            if user_prompt and assistant_response:
                chat = [{"role" : "user", "content" : f"{sys_prompt}\n\n{user_prompt}"},
                        {"role": "assistant", "content": assistant_response}]
            else:
                chat = [{"role" : "user", "content" : f"{sys_prompt}"}]
    else:
        if user_prompt and assistant_response:
            chat += f"\n\n{user_prompt}\n\n{assistant_response}"
    return chat

def build_few_shot_examples(examples, sys_prompt, user_prompt_template, assistant_response_template, has_system, is_instruct):
    for i, example in enumerate(examples):
            user_prompt = user_prompt_template.format(**example)
            assistant_response = assistant_response_template.format(**example)
            if i == 0:
                chat = build_first_turn(sys_prompt, user_prompt, assistant_response, has_system, is_instruct)
            else:
                if is_instruct:
                    chat += [{"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": assistant_response}]
                                
                else:
                    chat += f"\n\n{user_prompt}\n\n{assistant_response}"
    return chat

def build_model_input(example, user_prompt_template, model_is_instruct, few_shot_chat, tokenizer):
    user_prompt = user_prompt_template.format(**example)
    if model_is_instruct:
        chat = [{"role": "user", "content": f"{user_prompt}"}]
        if few_shot_chat:
            chat = few_shot_chat + chat
        for msg in chat:
            if isinstance(msg, list):
                print(msg)
        model_input = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True)

    else:
        model_input = f"\n\n{user_prompt}\n\n"
        if few_shot_chat:
            model_input = few_shot_chat + model_input

    return model_input


def get_UMLS_entities(doc): 
    global nlp, linker
    if nlp is None:
        import spacy
        import scispacy
        from scispacy.linking import EntityLinker
        nlp = spacy.load("en_core_sci_sm")
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        print("Loading scispacy UMLS Linker...")
        linker = nlp.get_pipe("scispacy_linker")

    entities = set()
    for entity in doc.ents:
        if entity._.kb_ents:
            entities.add(linker.kb.cui_to_entity[entity._.kb_ents[0][0]].canonical_name)
    return entities

def compute_UMLS_F1(model_output, label):
    global nlp, linker
    if nlp is None:
        import spacy
        import scispacy
        from scispacy.linking import EntityLinker
        nlp = spacy.load("en_core_sci_sm")
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        print("Loading scispacy UMLS Linker...")
        linker = nlp.get_pipe("scispacy_linker")
    
    
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


def update_results(results, new_results):
    for k in new_results:
        if k not in results:
            results[k] = [new_results[k]]
        results[k].append(new_results[k])
    return None

def compute_average_results(results):
    average_results = {}
    for k in results:
        average_results[k] = mean(results[k])
    return average_results


def compute_icd_f1(preds, gt, approximate):
    if approximate:
        preds = set([pred[:3] for pred in preds])
        gt = set([g[:3] for g in gt])
    
    if len(preds) == 0:
        P = 0.0
    else:
        P = len([pred for pred in preds if pred in gt]) / len(preds)
    R = len([g for g in gt if g in preds]) / len(gt)

    if (P + R) == 0:
        F = 0.0
    else:
        F = 2 * P * R / (P + R)

    return P, R, F

def is_icd10_valid(code):
    return icd10.find(code) is not None

def parse_icd_codes(text):
    # Regex to match ICD-10 codes
    pattern = r'\b[A-TV-Z][0-9]{2}(?:\.?[0-9A-Z]{1,4})?\b'

    # Find all matches
    return re.findall(pattern, text)
