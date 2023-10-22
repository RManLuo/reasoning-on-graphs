# Reasoning on Graphs (RoG)

Official Implementation of "[Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning](https://arxiv.org/abs/2310.01061)".

<img src="resources/rog.png" width = "800" />

Reasoning on graphs (RoG) synergizes LLMs with KGs to enable faithful and interpretable reasoning. We present a planning-retrieval-reasoning framework, where RoG first generates relation paths grounded by KGs as faithful plans. These plans are then used to retrieve valid reasoning paths from the KGs for LLMs to conduct faithful reasoning and generate interpretable results.

## Requirements

```
pip install -r requirements.txt
```

## Pre-trained weights

You can find the pre-trained weights [here](https://huggingface.co/rmanluo/RoG).

## Datasets

[RoG-WebQSP](https://huggingface.co/datasets/rmanluo/RoG-webqsp)
[RoG-CWQ](https://huggingface.co/datasets/rmanluo/RoG-cwq)

## Inference

### Step1: Planning (Generate relation paths)

Run: `./scripts/planning.sh`

```bash
python src/qa_prediction/gen_rule_path.py \
        --model_name RoG \
        --model_path rmanluo/RoG \
        -d {RoG-webqsp,RoG-cwq} \
        --split test \
        --n_beam 3
```

Generated rules will be saved at: `results/gen_rule_path/{dataset}/{model_name}/{split}`

### Step2: Reasoning (Generate answers with RoG)

Run: `./scripts/rog-reasoning.sh`

```bash
python src/qa_prediction/predict_answer.py \
        --model_name RoG \
        --model_path rmanluo/RoG \
        -d {RoG-webqsp,RoG-cwq} \
        --prompt_path prompts/llama2_predict.txt \
        --add_rul \
        --rule_path {rule_path} \
```

Answers will be saved at: `results/KGQA/{dataset}/{model_name}/{split}`

### Plug-and-play Reasoning (Generate answers with different LLMs)
>
> Note: you need to set your openai key at `.env` to use ChatGPT.

Run: `./scripts/plug-and-play.sh`

```bash
python src/qa_prediction/predict_answer.py \
        --model_name {gpt-3.5-turbo,alpaca,llama2-chat-hf,flan-t5} \
        -d {RoG-webqsp,RoG-cwq} \
        --prompt_path {prompt_path} \
        --add_rule \
        --rule_path {rule_path}
```
### Interpretable Reasoning
Run: `python scripts/interpretable_example.py`

```python
from transformers import pipeline, AutoTokenizer
import torch

MODEL_PATH_OR_NAME="rmanluo/RoG"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_OR_NAME, use_fast=False)
model = pipeline("text-generation", model=MODEL_PATH_OR_NAME, tokenizer=tokenizer, device_map="auto", torch_dtype=torch.float16)

print("====EXAMPLE 1: ====")

INPUT_TEXT_1 = """Based on the reasoning paths, please answer the given question and explain why 

Reasoning Paths: 
Northern District -> location.administrative_division.first_level_division_of -> Israel -> government.form_of_government.countries -> Parliamentary system

Question: 
What type of government is used in the country with Northern District?"""

outputs = model(INPUT_TEXT_1, return_full_text=False)
print(outputs[0]['generated_text'])
```

## Training

Training code will be available soon.

## Results

<img src="resources/results.png" width = "600" />
<img src="resources/plug-and-play.png" width = "600" />
<img src="resources/lack_of_knowledge.png" width = "600" />
<img src="resources/hallucination.png" width = "600" />

## Bibinfo
If you found this repo helpful, please help us by citing this paper:
```
@article{luo_rog,
title={Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning},
author={Luo, Linhao and Li, Yuan-Fang and Haffari, Gholamreza and Pan, Shirui},
journal={arXiv preprint arxiv:2310.01061},
year={2023}
}
```
