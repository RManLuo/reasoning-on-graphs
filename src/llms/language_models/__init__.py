from .chatgpt import ChatGPT
from .alpaca import Alpaca
from .longchat.longchat import Longchat
from .base_language_model import BaseLanguageModel
from .llama import Llama
from .flan_t5 import FlanT5

registed_language_models = {
    'gpt-4': ChatGPT,
    'gpt-3.5-turbo': ChatGPT,
    'alpaca': Alpaca,
    'longchat': Longchat,
    'llama': Llama,
    'flan-t5': FlanT5,
    'rog': Llama,
}

def get_registed_model(model_name) -> BaseLanguageModel:
    for key, value in registed_language_models.items():
        if key in model_name.lower():
            return value
    raise ValueError(f"No registered model found for name {model_name}")
