MODEL_PATH=meta-llama/Llama-2-7b-chat-hf
DATASET_LIST="datasets/joint_training/align/cwq/cwq_train.jsonl datasets/joint_training/align/webqsp/webqsp_train.jsonl datasets/joint_training/qa/webqsp/webqsp_train.jsonl datasets/joint_training/qa/cwq/cwq_train.jsonl datasets/joint_training/ExplainQAData/cwq/cwq_train_1000.jsonl datasets/joint_training/ExplainQAData/webqsp/webqsp_train_1000.jsonl"
SAVE_NAME=RoG
SAVE_PATH=save_models/${SAVE_NAME}
ADD_REL=False

accelerate launch --config_file config/deepspeed_zero3.yml src/joint_training/joint_finetuning.py \
    --data_path_list ${DATASET_LIST}  \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${SAVE_PATH} \
    --add_rel_token ${ADD_REL} \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --report_to "wandb" \
    --gradient_checkpointing True \
    --run_name ${SAVE_NAME}"