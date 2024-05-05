from pathlib import Path
import json

import pandas as pd



models = ["Llama3-OpenBioLLM-8B", "Llama3-OpenBioLLM-70B", "Meta-Llama-3-8B-Instruct", "Meta-Llama-3-70B-Instruct", "Mixtral-8x22B-Instruct-v0.1", "Mixtral-8x7B-Instruct-v0.1", "BioMistral-7B-DARE", "zephyr-7b-beta", "Mistral-7B-Instruct-v0.1"]
results = {}
for model in models:
    for result_path in Path("predictions").glob(f"*/{model}/results.json"):
        dataset = result_path.parents[1].name
        with open(result_path, "r") as f:
            model_results = json.loads(f.read())
        if dataset not in results:
            results[dataset] = {}
        if model not in results[dataset]:
            results[dataset][model] = {}
        
        for metric in model_results:
            if "_P" in metric or "_R" in metric or " P" in metric or " R" in metric:
                continue
            results[dataset][model][metric] = f"{model_results[metric]:.2%}"[:-1]
        
        print()



for dataset in results:
    pd.DataFrame(results[dataset]).transpose().to_csv(f"results/{dataset}_results.csv")
        