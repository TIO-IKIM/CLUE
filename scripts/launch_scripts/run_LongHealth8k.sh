#!/bin/bash

# Tasks specific settings
eval_script="eval/eval_LongHealth.py"
dataset="data/LongHealth/benchmark_v5.json"
max_len="8140"
prediction_path="predictions/LongHealth8k"



# Default values for flags
model_has_system=false
model_is_instruct=false

# Extract parameters using named flags
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --model_address) model_address="$2"; shift ;;
        --model_id) model_id="$2"; shift ;;
        --model_has_system) model_has_system=true ;;
        --model_is_instruct) model_is_instruct=true ;;
        --log_path) log_path="$2"; shift ;;
        --token) token="$2"; shift ;;
        *) echo "Unknown parameter: $1" ;;
    esac
    shift
done

# Paths and script specific parameters
[[ -z "$log_path" ]] && log_path="$prediction_path/$(basename "$model_id")"

# Command construction
cmd="python $eval_script --model_address $model_address --model_name_or_path $model_id"
[[ $model_has_system == true ]] && cmd+=" --model_has_system"
[[ $model_is_instruct == true ]] && cmd+=" --model_is_instruct"
cmd+=" --data_path $dataset --log_path $log_path --max_len $max_len"
[[ -n $token ]] && cmd+=" --token $token"

# Execute the command
eval $cmd
