model_address=$1
model_id=$2


for script in "scripts/launch_scripts"/*; do
    # Check if the script exists and is executable
    if [ -x "$script" ]; then
      echo "Executing: $script with $model_id running at $model_address"
      # Execute the script
      "$script" $model_address $model_id
    else
      echo "Script not found or not executable: $script"
    fi
done