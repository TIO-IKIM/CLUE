eval_script="eval/eval_BioNLP2023.py"
dataset="data/BioNLP2023/BioNLP2023-1A-Test.csv"
log_path="predictions/BioNLP2023"
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