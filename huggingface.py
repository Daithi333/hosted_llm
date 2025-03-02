from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class HuggingFace:

    @staticmethod
    def save_model(model_name: Any, path: str):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        model.save_pretrained(path)

    @staticmethod
    def save_tokenizer(model_name: Any, path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(path)
