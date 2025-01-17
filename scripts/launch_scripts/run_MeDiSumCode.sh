#!/bin/bash

# Tasks specific settings
eval_script="eval/eval_MeDiSumCode.py"
dataset="data/MeDiSumCode/MeDiSumCode.json"
prediction_path="predictions/MeDiSumCode"
num_few_shot_examples=1


# Extract parameters using named flags
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --model) model="$2"; shift ;;
        --model_has_system) model_has_system=true ;;
        --model_is_instruct) model_is_instruct=true ;;
        --log_path) log_path="$2"; shift ;;
        --token) token="$2"; shift ;;
        *) echo "Unknown parameter: $1" ;;
    esac
    shift
done

# Paths and script specific parameters
[[ -z "$log_path" ]] && log_path="$prediction_path/$(basename "$model")"

# Command construction
cmd="uv run $eval_script --model $model"
cmd+=" --num_few_shot_examples $num_few_shot_examples --data_path $dataset --log_path $log_path"
[[ -n $token ]] && cmd+=" --token $token"

# Execute the command
eval $cmd
