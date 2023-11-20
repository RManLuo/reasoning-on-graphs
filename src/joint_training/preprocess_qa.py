import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils import *
from transformers import AutoTokenizer
import datasets
from qa_prediction.build_qa_input import PromptBuilder

N_CPUS = int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else 1

save_dir = "datasets/joint_training/qa"
prompt_path = "prompts/llama2_predict.txt"
split="train"
model_max_length = 2048 - 200
data_list = ['RoG-webqsp', 'RoG-cwq']
data_path = "rmanluo"
model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
prompter = InstructFormater(prompt_path)

tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )


# Load prompt template
input_builder = PromptBuilder(
        prompt_path,
        add_rule = True,
        use_true= True,
        maximun_token=model_max_length,
        tokenize=lambda x: len(tokenizer.tokenize(x)),
    )

def formatting_prompts_func(example):
    output_label = "\n".join(example['answer'])
     # Find ground-truth paths for each Q-P pair
    graph = build_graph(example["graph"])
    paths = get_truth_paths(example["q_entity"], example["a_entity"], graph)
    ground_paths = set()
    for path in paths:
        ground_paths.add(tuple([p[1] for p in path]))  # extract relation path
    example["ground_paths"] = list(ground_paths)
    output_text = (
            input_builder.process_input(example)
            + " "
            + output_label + tokenizer.eos_token
        )
    return {"text": output_text}

for data_name in data_list:
    input_file = os.path.join(data_path, data_name)
    train_dataset = datasets.load_dataset(input_file, split="train")
    save_path = os.path.join(save_dir, data_name, data_name + "_train.jsonl")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    # with open(save_path, "w") as f:
    #     print("Processing {}...".format(data_name))
    #     print("Number of process: {}".format(N_CPUS))
    #     with mp.Pool(N_CPUS) as pool:
    #         for example in tqdm(pool.imap_unordered(formatting_prompts_func, train_dataset), total=len(train_dataset)):
    #             f.write(json.dumps(example) + "\n")
    train_dataset = train_dataset.map(
        formatting_prompts_func,
        remove_columns=train_dataset.column_names,
        num_proc=N_CPUS,
    )
    train_dataset.to_json(save_path, orient="records", lines=True)
