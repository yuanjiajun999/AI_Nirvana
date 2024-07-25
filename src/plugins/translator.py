from transformers import MarianMTModel, MarianTokenizer


class TranslatorPlugin:
    def __init__(self):
        self.model_name = "Helsinki-NLP/opus-mt-en-zh"
        self.model = MarianMTModel.from_pretrained(self.model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)

    def execute(
        self, text: str, source_lang: str = "en", target_lang: str = "zh"
    ) -> str:
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        output = self.model.generate(input_ids)
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded


def register_plugin():
    return TranslatorPlugin()
