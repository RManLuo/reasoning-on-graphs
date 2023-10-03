from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from ..base_language_model import BaseLanguageModel

def maybe_monkey_patch(args):
    from .llama_condense_monkey_patch import replace_llama_with_condense
    replace_llama_with_condense(args.longchat_ratio)

    if args.longchat_flash_attn:
        from .llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()
        

class Longchat(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--model_path', type=str, help="HUGGING FACE MODEL or model path", default='lmsys/longchat-7b-16k')
        parser.add_argument('--max_new_tokens', type=int, help="max length", default=512)
        parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')
        parser.add_argument("--longchat_flash_attn", action='store_true', help="Only apply to longchat models. Whether to enable flash attention to save memory, but slower.")
        parser.add_argument("--longchat_ratio", type=int, default=8, help="Only apply to longchat models. Use ratio=8 for 16K context length model. Only ratio=8 is supported now.")

    def __init__(self, args):
        self.args = args
        self.maximun_token = 16384 - 100
        
    
    def load_model(self, **kwargs):
        maybe_monkey_patch(self.args)
        model = AutoModelForCausalLM.from_pretrained(**kwargs)
        return model
    
    def tokenize(self, text):
        return len(self.tokenizer.tokenize(text))
    
    def prepare_for_inference(self, **model_kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, use_fast=False)
        maybe_monkey_patch(self.args)
        self.generator = pipeline("text-generation", model=self.args.model_path, tokenizer=self.tokenizer, device_map="auto", torch_dtype=self.DTYPE.get(self.args.dtype, None), model_kwargs=model_kwargs)
        
    @torch.inference_mode()
    def generate_sentence(self, llm_input):
        outputs = self.generator(llm_input, return_full_text=False, max_new_tokens=self.args.max_new_tokens, use_cache=not self.args.longchat_flash_attn) # Flash Atten does not support cache
        return outputs[0]['generated_text'] # type: ignore