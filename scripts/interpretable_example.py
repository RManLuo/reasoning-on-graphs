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

print("====EXAMPLE 2: ====")

INPUT_TEXT_2 = """Based on the reasoning paths, please answer the given question and explain why.

Reasoning Paths:
1946 World Series -> sports.sports_team.championships -> St. Louis Cardinals -> sports.sports_team.arena_stadium -> Busch Stadium
1946 World Series -> sports.sports_team.championships -> St. Louis Cardinals -> sports.sports_team.arena_stadium -> Roger Dean Stadium

Question:
Where is the home stadium of the team who won the 1946 World Series championship?"""

outputs = model(INPUT_TEXT_2, return_full_text=False)

print(outputs[0]['generated_text'])