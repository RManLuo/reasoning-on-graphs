SPLIT="test"
DATASET_LIST="RoG-webqsp RoG-cwq"
MODEL_NAME=RoG
MODEL_PATH=rmanluo/RoG

BEAM_LIST="3" # "1 2 3 4 5"
for DATASET in $DATASET_LIST; do
    for N_BEAM in $BEAM_LIST; do
        python src/qa_prediction/gen_rule_path.py \
        --model_name ${MODEL_NAME} \
        --model_path ${MODEL_PATH} \
        -d ${DATASET} \
        --split ${SPLIT} \
        --n_beam ${N_BEAM}
    done
done