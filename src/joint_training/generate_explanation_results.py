import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils import *
import datasets
from qa_prediction.build_qa_input import PromptBuilder
from transformers import AutoTokenizer
from llms.language_models import ChatGPT
from collections import namedtuple
import multiprocessing as mp
from tqdm import tqdm

N_CPUS = int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else 1

save_dir = "datasets/joint_training/ExplainQAData"
split="train"
model_max_length = 1024
data_list = ['webqsp', 'cwq']
data_path = "/home/lluo/projects/KIT/data/KGQA"

Config = namedtuple("Config", "retry model_name")
config = Config(retry=3, model_name = "gpt-3.5-turbo")
model = ChatGPT(config)
gpt_prompt_path = "prompts/general_prompt.txt"
model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )

few_shot_examples = """## Input:
Based on the reasoning paths, please answer the given question and explain why 

Reasoning Paths: 
Northern District -> location.administrative_division.first_level_division_of -> Israel -> government.form_of_government.countries -> Parliamentary system

Question: 
What type of government is used in the country with Northern District?

## Output:
Answer:
Parliamentary system

Explanation:
1. "Northern District" is a location within some country.
2. The reasoning path mentions "Northern District -> location.administrative_division.first_level_division_of -> Israel," indicating that the Northern District is part of Israel.
3. It further states "Israel -> government.form_of_government.countries," suggesting that Israel's form of government is being discussed.
4. The last part of the reasoning path indicates that Israel has a "Parliamentary system."

Therefore, based on the provided reasoning paths, it can be concluded that the type of government used in the country with the Northern District (Israel) is a Parliamentary system.

## Input:
Based on the reasoning paths, please answer the given question and explain why.

Reasoning Paths:
1946 World Series -> sports.sports_team.championships -> St. Louis Cardinals -> sports.sports_team.arena_stadium -> Busch Stadium
1946 World Series -> sports.sports_team.championships -> St. Louis Cardinals -> sports.sports_team.arena_stadium -> Roger Dean Stadium

Question:
Where is the home stadium of the team who won the 1946 World Series championship?

## Output:
Answer:
Busch Stadium

Explanation:
1. 1946 World Series -> sports.sports_team.championships -> St. Louis Cardinals -> sports.sports_team.arena_stadium -> Busch Stadium

The reasoning path leads us to the St. Louis Cardinals as the team that won the 1946 World Series, and Busch Stadium is the stadium associated with the St. Louis Cardinals. Therefore, Busch Stadium is the home stadium of the team that won the 1946 World Series championship.

## Input:
Based on the reasoning paths, please answer the given question and explain why.

Reasoning Paths:
Lou Seal -> sports.mascot.team -> San Francisco Giants -> sports.sports_championship_event.champion -> 2014 World Series

Question:
Lou Seal is the mascot for the team that last won the World Series when?

## Output:
Answer:
Lou Seal is the mascot for the team that last won the World Series in 2014.

Explanation:
1. The reasoning path starts with "Lou Seal" and links it to "sports.mascot.team."
2. From there, it leads to "San Francisco Giants," indicating that Lou Seal is the mascot for the San Francisco Giants.
3. The path then continues to "sports.sports_championship_event.champion -> 2014 World Series," which tells us that the San Francisco Giants were the champions of the 2014 World Series.

Therefore, based on the provided reasoning paths, it can be concluded that the San Francisco Giants, represented by Lou Seal, last won the World Series in 2014.

## Input:
"""


# Load prompt template
gpt_builder = PromptBuilder(
        gpt_prompt_path,
        add_rule = True,
        use_true= True,
        explain=True,
        maximun_token=model_max_length,
        tokenize=model.tokenize,
    )

def formatting_prompts_func(example):
    output_label = "\n".join(example['answer'])
     # Find ground-truth paths for each Q-P pair
    graph = build_graph(example["graph"])
    paths = get_truth_paths(example["q_entity"], example["a_entity"], graph)
    ground_paths = set()
    for path in paths:
        ground_paths.add(tuple([p[1] for p in path]))  # extract relation path
    if len(ground_paths) == 0:
        return None
    example["ground_paths"] = list(ground_paths)
    input_text = gpt_builder.process_input(example)
    # Filter empty
    if "Reasoning Paths:\n\n\n" in input_text:
        return None
    few_shot_input = few_shot_examples + input_text
    prediction = model.generate_sentence(few_shot_input)
    if prediction is not None:
        output_text = "[INST] <<SYS>>\n<</SYS>>\n" + input_text + ' [/INST] ' + prediction + tokenizer.eos_token
        return {"text": output_text}
    else:
        return None

for data_name in data_list:
    input_file = os.path.join(data_path, data_name)
    train_dataset = datasets.load_dataset(input_file, split="train")
    train_dataset = train_dataset.shuffle().select(range(1000))
    save_path = os.path.join(save_dir, data_name, data_name + "_train_1000.jsonl")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, "w") as f:
        print("Processing {}...".format(data_name))
        print("Number of process: {}".format(N_CPUS))
        with mp.Pool(N_CPUS) as pool:
            for example in tqdm(pool.imap_unordered(formatting_prompts_func, train_dataset), total=len(train_dataset)):
                if example is not None:
                    f.write(json.dumps(example) + "\n")
    # train_dataset = train_dataset.map(
    #     formatting_prompts_func,
    #     remove_columns=train_dataset.column_names,
    #     num_proc=N_CPUS,
    # )
    # train_dataset.to_json(save_path, orient="records", lines=True)
