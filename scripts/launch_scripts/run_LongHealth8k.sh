#!/bin/bash

# Tasks specific settings
eval_script="eval/eval_LongHealth.py"
dataset="data/LongHealth/benchmark_v5.json"
max_len="8140"
prediction_path="predictions/LongHealth8k"


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
cmd="uv run $eval_script --model $model"
cmd+=" --data_path $dataset --log_path $log_path --max_len $max_len"
[[ -n $token ]] && cmd+=" --token $token"

# Execute the command
eval $cmd
