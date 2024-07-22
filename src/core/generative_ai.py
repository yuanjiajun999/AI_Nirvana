import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GenerativeAI:
    def __init__(self, model_path):
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    def generate_text(self, prompt, max_length=50, num_return_sequences=1):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_beams=2,
        )
        generated_texts = [self.tokenizer.decode(gen_text, skip_special_tokens=True) for gen_text in output]
        return generated_texts