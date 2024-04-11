eval_script="eval/eval_MeDiSumQA.py"
dataset="data/MeDiSumQA/MeDiSumQA.json"
log_path="predictions/MeDiSumQA"
model_address=$1
model_id=$2
model_name="$(basename $model_id)"

python $eval_script \
    --model_address $1 \
    --model_name_or_path $2 \
    --num_few_shot_examples 1 \
    --model_is_instruct \
    --data_path $dataset \
    --log_path $log_path/model_name