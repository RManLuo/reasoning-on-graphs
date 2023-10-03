SPLIT="test"
DATASET_LIST="RoG-webqsp RoG-cwq"
MODEL_NAME=RoG
PROMPT_PATH=prompts/llama2_predict.txt
BEAM_LIST="3" # "1 2 3 4 5"

for DATA_NAME in $DATASET_LIST; do
    for N_BEAM in $BEAM_LIST; do
        RULE_PATH=results/gen_rule_path/${DATA_NAME}/${MODEL_NAME}/test/predictions_${N_BEAM}_False.jsonl
        python src/qa_prediction/predict_answer.py \
            --model_name ${MODEL_NAME} \
            -d ${DATA_NAME} \
            --prompt_path ${PROMPT_PATH} \
            --add_rule \
            --rule_path ${RULE_PATH} \
            --model_path rmanluo/RoG
    done
done