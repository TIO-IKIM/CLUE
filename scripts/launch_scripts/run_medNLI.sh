eval_script="eval/eval_mednli.py"
dataset="data/MedNLI/mli_test_v1.jsonl"
log_path="predictions/MedNLI"
model_address=$1
model_id=$2
model_name="$(basename $model_id)"

python $eval_script \
    --model_address $1 \
    --model_name_or_path $2 \
    --num_few_shot_examples 3 \
    --model_is_instruct \
    --data_path $dataset \
    --log_path $log_path/model_name