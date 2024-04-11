eval_script="eval/eval_meqsum.py"
dataset="data/MeQSum/MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx"
log_path="predictions/MeQSum"
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