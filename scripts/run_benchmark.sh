#!/bin/bash

# Initialize variables
model_address=""
model_id=""
model_has_system=false
model_is_instruct=false
log_path=""
token=""

# Usage information
usage() {
    echo "Usage: $0 --model_address <local-model-address> --model_id <hf-model-id> [OPTIONS]"
    echo "Options:"
    echo "  --model_has_system           Indicate if the model has a system prompt."
    echo "  --model_is_instruct          Indicate if the model is instruction tuned."
    echo "  --log_path <path>            Specify the log folder path."
    echo "  --token <API token>          Specify the API token."
    exit 1
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_address) model_address="$2"; shift ;;
        --model_id) model_id="$2"; shift ;;
        --model_has_system) model_has_system=true ;;
        --model_is_instruct) model_is_instruct=true ;;
        --log_path) log_path="$2"; shift ;;
        --token) token="$2"; shift ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

# Check mandatory parameters
if [[ -z "$model_address" || -z "$model_id" ]]; then
    echo "Error: --model_address and --model_id are required."
    usage
fi

# Iterate through scripts in the specified directory
for script in "scripts/launch_scripts"/*; do
    # Check if the script exists and is executable
    if [ -x "$script" ]; then
        echo "Executing: $script with --model_id $model_id running at --model_address $model_address"
        # Prepare command with base required arguments
        cmd="$script --model_address $model_address --model_id $model_id"
        [[ $model_has_system == true ]] && cmd+=" --model_has_system"
        [[ $model_is_instruct == true ]] && cmd+=" --model_is_instruct"
        [[ -n $log_path ]] && cmd+=" --log_path $log_path"
        [[ -n $token ]] && cmd+=" --token $token"
        # Execute the command
        eval $cmd
    else
        echo "Script not found or not executable: $script"
    fi
done