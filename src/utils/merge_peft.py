from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
import torch
from peft import AutoPeftModelForCausalLM

@dataclass
class ScriptArguments:
    input_path: Optional[str] = field(default=None, metadata={"help": "input path"})
    output_path: Optional[str] = field(default=None, metadata={"help": "output the path"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
model = AutoPeftModelForCausalLM.from_pretrained(script_args.input_path, device_map="auto", torch_dtype=torch.bfloat16)
model = model.merge_and_unload()
model.eval()
model.save_pretrained(script_args.output_path)