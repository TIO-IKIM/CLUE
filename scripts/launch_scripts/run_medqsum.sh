#!/bin/bash

# Tasks specific settings
eval_script="eval/eval_meqsum.py"
dataset="data/MeQSum/MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx"
prediction_path="predictions/MeQSum"
num_few_shot_examples=3

# Extract parameters using named flags
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --model) model="$2"; shift ;;
        --log_path) log_path="$2"; shift ;;
        --token) token="$2"; shift ;;
        *) echo "Unknown parameter: $1" ;;
    esac
    shift
done

# Paths and script specific parameters
[[ -z "$log_path" ]] && log_path="$prediction_path/$(basename "$model")"

# Command construction
cmd="uv run $eval_script $model_address --model $model"
cmd+=" --num_few_shot_examples $num_few_shot_examples --data_path $dataset --log_path $log_path"
[[ -n $token ]] && cmd+=" --token $token"

# Execute the command
eval $cmd