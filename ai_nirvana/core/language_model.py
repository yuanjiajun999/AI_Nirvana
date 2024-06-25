from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LanguageModel:
    def __init__(self, model_name="gpt2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_response(self, prompt, context=None, max_length=150):
        if context:
            prompt = f"{context}\n\nHuman: {prompt}\nAI:"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones(inputs.shape, device=self.device)
        
        outputs = self.model.generate(
            inputs, 
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)