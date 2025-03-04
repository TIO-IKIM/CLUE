# CLUE: A Clinical Language Understanding Evaluation for LLMs

[Project Website](https://clue.ikim.nrw/) | [Paper](https://arxiv.org/abs/2404.04067)

CLUE is a benchmark to evaluate the clinical language understanding of LLMs. It consists of 6 tasks, including two novel ones based on MIMIC IV notes. This repository provides the code to run the benchmark and generate the new tasks.

<p align="center">
  <img src="images/CLUE_overview.png" width="300"/>
</p>

## Updates
- [2025/01/17] The code base is now build on [uv](https://github.com/astral-sh/uv) and [vLLM](https://github.com/vllm-project/vllm) simplifying the setup considerably.
- [2024/06/25] We released an updated version of our [preprint](https://arxiv.org/abs/2404.04067) with an evaluation of 25 LLMs. Additionally, we included manual quality checks for MeDiSumQA. This refined dataset will be released on PhysioNet. The new results can also be found on the [project website](https://clue.ikim.nrw/).

## Contents
- [Motivation](#motivation)
- [Benchmark Compilation](#benchmark-compilation)
- [Run Evaluation](#run-evaluation)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Motivation

Despite the advancements promised by biomedical LLMs for patient care, a significant gap exists in their evaluation, particularly concerning their application in real-world clinical settings. Existing assessments, focused on medical knowledge through constructed questions, fall short of capturing the complexity and diversity of clinical tasks. Additionally, the rapid pace at which LLMs evolve further complicates selecting the most appropriate models for healthcare applications. In response to these challenges, CLUE aims to offer a comprehensive and standardized framework for assessing the performance of both specialized biomedical and advanced general-domain LLMs in practical healthcare tasks.

## Results

<p align="center">
  <img src="images/CLUE_results.png" width="100%"/>
</p>

**Top Performers**: General-domain models achieved the highest scores overall. Biomedical models generally performed lower, with some scoring below the baseline, indicating adaptation challenges.

**Performance Differences**: High-performing models showed smaller score differences at level 1 compared to level 2 tasks, demonstrating the expressiveness of the new tasks MeDiSumQA and MeDiSumCode and the LongHealth dataset.

**Effect of Biomedical Fine-Tuning**: Biomedical fine-tuning often led to performance declines, with many models experiencing significant drops. Instruction-tuned general-domain models consistently outperformed biomedical models. Improvements in some biomedical models were mainly in those based on earlier versions of general-domain models, indicating recent advancements may have already addressed some gaps fine-tuning aimed to fill. In conclusion, current biomedical training methods and datasets do not significantly enhance model performance in clinical settings. Future research should focus on novel training methods and specialized datasets to address these issues.


## Benchmark Compilation
We introduce two novel tasks based on MIMIC IV discharge summaries. The following section describes how to generate these tasks and collect the existing ones.

### MeDiSumQA
To run the data generation pipeline described in the paper, a few preparatory steps are necessary:

- You need access to [MIMIC-IV-Note v2.2](https://physionet.org/content/mimic-iv-note/2.2/)
- A locally running LLM. Our scripts are compatible with [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) hosted with the [Text Generation Inference](https://github.com/huggingface/text-generation-inference). For any other configuration, please adjust the api calls and tokenizer accordingly.
- Install the dependencies for this repo from the requirements.txt file.

The following commands are generating the MeDiSumQA dataset:


We provide a [1-shot-example](data/MeDiSumQA/example_statements.txt) for the statement extraction prompt. To retrieve the corresponding discharge letter run:
```bash
python3 data/MeDiSumQA/get_example.py --mimic_discharge_notes <path-to-mimic-notes>
```
This should create a file data/MeDiSumQA/example_summary.txt with the discharge summary from which the statements are extracted.

To generate the question-answer pairs you can now run:


```bash
python3 data/MeDiSumQA/generate_qas.py \
    --mimic_discharge_notes <path-to-mimic-notes> \
    --max_discharge_notes 4000 \
    --example_ds data/MeDiSumQA/example_summary.txt \
    --example_ds_statements data/MeDiSumQA/example_statements.txt \
    --output_file data/MeDiSumQA/MeDiSumQA_raw_qas.json \
    --model_address <local-url-to-your-model>
```

The final step is to compute the BERTScore and ROUGE-score between the generated questions and answers and filter out the ones with the lowest similarity.

```bash
python3 data/MeDiSumQA/post_process_qas.py \
    --dataset data/MeDiSumQA/MeDiSumQA_raw_qas.json \
    --top_k_qas 500 \
    --output_path data/MeDiSumQA/MeDiSumQA.json

```

### MeDiSumCode

To generate MeDiSumCode you need access to the MIMIC IV v2.2 hosp module. Specifically the file [diagnoses_icd.csv.gz](https://physionet.org/content/mimiciv/2.2/hosp/diagnoses_icd.csv.gz).

The following script generates the dataset:

```bash
python3 data/MeDiSumCode/generate_code_dataset.py \
    --mimic_diagnoses_icd <path-to-mimic-diagnoses-icd>
    --mimic_discharge_notes <path-to-mimic-notes> \
    --output_file data/MeDiSumCode/MeDiSumCode.json \
    --max_examples 500
```



### Other Datasets

All other datasets can be downloaded with this command:

```bash

./scripts/download_datasets.sh <your-physionet-username>
```

Make sure you have access to the resources on physionet.

## Run Evaluation


### Basic Usage

```bash
./scripts/run_clue.sh --model <hf-model-id> [OPTIONS]
```
### Arguments

- --model <hf-model-id>: Identifier for the model registered on Hugging Face.

### Options
- --log_path <path>: Specifies the directory for log files. Defaults to the model name if not provided.
- --token <API token>: HF Hub API token required for authentication.

## Acknowledgement

This benchmark was made possible by the provision of multiple datasets:

- [MedNLI](https://jgc128.github.io/mednli/)
- [MeQSum](https://github.com/abachaa/MeQSum)
- [Problem List Summarization](https://physionet.org/content/bionlp-workshop-2023-task-1a/2.0.0/)
- [LongHealth](https://github.com/kbressem/LongHealth)
- [MIMIC IV](https://physionet.org/content/mimiciv/2.2/)
  
We sincerely thank the respective authors for allowing us this opportunity.

## Citation

```bibtex
@misc{dada2024clue,
      title={CLUE: A Clinical Language Understanding Evaluation for LLMs}, 
      author={Amin Dada and Marie Bauer and Amanda Butler Contreras and Osman Alperen Koraş and Constantin Marc Seibold and Kaleb E Smith and Jens Kleesiek},
      year={2024},
      eprint={2404.04067},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
