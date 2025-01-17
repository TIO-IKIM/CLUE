#!/bin/bash

# Initialize variables
model=""
log_path=""
token=""

# Usage information
usage() {
    echo "Usage: $0 --model <hf-model-id> [OPTIONS]"
    echo "Options:"
    echo "  --log_path <path>            Specify the log folder path."
    echo "  --token <API token>          Specify the API token."
    exit 1
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) model="$2"; shift ;;
        --log_path) log_path="$2"; shift ;;
        --token) token="$2"; shift ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

# Check mandatory parameters
if [[ -z "$model" ]]; then
    echo "Error: --model is required."
    usage
fi

# Iterate through scripts in the specified directory
for script in "scripts/launch_scripts"/*; do
    # Check if the script exists and is executable
    if [ -x "$script" ]; then
        echo "Executing: $script with $model"
        # Prepare command with base required arguments
        cmd="$script --model $model"
        [[ -n $log_path ]] && cmd+=" --log_path $log_path"
        [[ -n $token ]] && cmd+=" --token $token"
        # Execute the command
        eval $cmd
    else
        echo "Script not found or not executable: $script"
    fi
done