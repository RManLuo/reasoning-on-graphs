SPLIT="test"
DATASET_LIST="RoG-webqsp RoG-cwq"
BEAM_LIST="3" # "1 2 3 4 5"
MODEL_LIST="gpt-3.5-turbo alpaca llama2-chat-hf flan-t5"
PROMPT_LIST="prompts/general_prompt.txt prompts/alpaca.txt prompts/llama2_predict.txt prompts/general_prompt.txt"
set -- $PROMPT_LIST

for DATA_NAME in $DATASET_LIST; do
    for N_BEAM in $BEAM_LIST; do
        RULE_PATH=results/gen_rule_path/${DATA_NAME}/${MODEL_NAME}/test/predictions_${N_BEAM}_False.jsonl
        for i in "${!MODEL_LIST[@]}"; do
        
            MODEL_NAME=${MODEL_LIST[$i]}
            PROMPT_PATH=${PROMPT_LIST[$i]}
            
            python src/qa_prediction/predict_answer.py \
                --model_name ${MODEL_NAME} \
                -d ${DATA_NAME} \
                --prompt_path ${PROMPT_PATH} \
                --add_rule \
                --rule_path ${RULE_PATH}
        done
    done
done