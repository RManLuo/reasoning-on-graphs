from transformers import pipeline, AutoModel, AutoTokenizer
import torch
from .base_language_model import BaseLanguageModel

class FlanT5(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    @staticmethod
    def add_args(parser):
        parser.add_argument('--model_path', type=str, help="HUGGING FACE MODEL or model path", default='google/flan-t5-xl')
        parser.add_argument('--max_new_tokens', type=int, help="max length", default=512)
        parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')

    def __init__(self, args):
        self.args = args
        self.maximun_token = 512 - 5
        
    def load_model(self, **kwargs):
        model = AutoModel.from_pretrained(**kwargs)
        return model
    
    def tokenize(self, text):
        return len(self.tokenizer.tokenize(text))
    
    def prepare_for_inference(self, **model_kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path,
        use_fast=False)
        self.generator = pipeline("text2text-generation", model=self.args.model_path, tokenizer=self.tokenizer, device_map="auto", model_kwargs=model_kwargs, torch_dtype=self.DTYPE.get(self.args.dtype, None))
    
    @torch.inference_mode()
    def generate_sentence(self, llm_input):
        outputs = self.generator(llm_input)
        return outputs[0]['generated_text'] # type: ignore