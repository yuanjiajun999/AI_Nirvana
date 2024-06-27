from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LanguageModel:
    def __init__(self, model_name="gpt2-large", use_gpu=False, system_prompt=""):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.system_prompt = system_prompt

    def generate_response(self, prompt, context=None, max_length=150):
        if context:
            full_prompt = f"{self.system_prompt}\n\nContext: {context}\n\nHuman: {prompt}\nAI:"
        else:
            full_prompt = f"{self.system_prompt}\n\nHuman: {prompt}\nAI:"

        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            input_ids,
            max_length=len(input_ids[0]) + max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        response = self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        return self.clean_response(response)

    def clean_response(self, response):
        # 移除可能的前缀
        prefixes = ["Human:", "AI:", "You:", "I:"]
        for prefix in prefixes:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()

        # 移除不完整的句子
        sentences = response.split('.')
        complete_sentences = '.'.join(sentences[:-1]) + ('.' if sentences[-1].strip() else '')

        return complete_sentences.strip()