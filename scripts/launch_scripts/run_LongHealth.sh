eval_script="eval/eval_LongHealth.py"
dataset="data/LongHealth/benchmark_v5.json"
log_path="predictions/LongHealth"
model_address=$1
model_id=$2
model_name="$(basename $model_id)"

python $eval_script \
    --model_address $1 \
    --model_name_or_path $2 \
    --model_is_instruct \
    --data_path $dataset \
    --log_path $log_path/model_name \
    --max_len 16000