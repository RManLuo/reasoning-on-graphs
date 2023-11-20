import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils import *
from transformers import AutoTokenizer
import datasets

N_CPUS = int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else 1

save_dir = "datasets/joint_training/align"
prompt_path = "prompts/llama2.txt"
data_template = "datasets/AlignData/{}/{}_train.jsonl"
data_list = ['RoG-webqsp', 'RoG-cwq']
model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
prompter = InstructFormater(prompt_path)

INSTRUCTION = """Please generate a valid relation path that can be helpful for answering the following question: """
SEP = '<SEP>'
BOP = '<PATH>'
EOP = '</PATH>'


tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )

def formatting_prompts_func(example):
        output_label = rule_to_string(example["path"], sep_token=SEP, bop=BOP, eop=EOP)
        output_text = (
                prompter.format(instruction=INSTRUCTION, message=example["question"])
                + " "
                + output_label + tokenizer.eos_token
            )
        return {"text": output_text}

for data_name in data_list:
    data_path = data_template.format(data_name, data_name)
    save_path = os.path.join(save_dir, data_name, data_name + "_train.jsonl")
    train_dataset = datasets.load_dataset('json', data_files=data_path, split="train")
    # if not os.path.exists(os.path.dirname(save_path)):
    #     os.makedirs(os.path.dirname(save_path))
    # with open(save_path, "w") as f:
    #     print("Processing {}...".format(data_name))
    #     print("Number of process: {}".format(N_CPUS))
    #     with mp.Pool(N_CPUS) as pool:
    #         for example in tqdm(pool.imap(formatting_prompts_func, train_dataset), total=len(train_dataset)):
    #             f.write(json.dumps(example) + "\n")
    
    train_dataset = train_dataset.map(
        formatting_prompts_func,
        remove_columns=["question", "path"],
        num_proc=N_CPUS,
    )
    train_dataset.to_json(save_path, orient="records", lines=True)
