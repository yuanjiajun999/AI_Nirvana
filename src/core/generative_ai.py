import os
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForImageClassification
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
import numpy as np
from typing import List, Union, Dict, Any, Optional
from src.utils.error_handler import error_handler, logger
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)  

def error_handler(func):  
    def wrapper(*args, **kwargs):  
        try:  
            return func(*args, **kwargs)  
        except Exception as e:  
            logger.error(f"Error in {func.__name__}: {str(e)}")  
            raise  
    return wrapper  

class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = item['input_ids'].clone()
        return item

class GenerativeAI:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("API_BASE")
        )
        self.model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo-0125")
        self.finetune_model_name = os.getenv("FINETUNE_MODEL_NAME", "distilgpt2")
        
        # 强制指定为CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化 image_classification_pipeline
        self.image_classification_pipeline = pipeline("image-classification", model="microsoft/resnet-50", device=0 if torch.cuda.is_available() else -1)
        # 初始化 image_captioning_pipeline
        self.image_captioning_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=0 if torch.cuda.is_available() else -1)
        
        # 加载微调模型
        self.finetune_model, self.tokenizer = self._load_finetune_model_and_tokenizer()

    @error_handler
    def _load_finetune_model_and_tokenizer(self, model_name=None):
        if model_name is None:
            model_name = self.finetune_model_name
        model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token  # 设置 pad_token
        return model, tokenizer
        
    @error_handler  
    def _load_model_and_tokenizer(self, model_name):  
        model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)  
        tokenizer = AutoTokenizer.from_pretrained(model_name)  
        return model, tokenizer  

    def generate_text(self, prompt, max_tokens=100, temperature=0.7, num_return_sequences=1, truncate=False):
        generated_texts = None
        try:
            if hasattr(self, 'finetune_model') and self.finetune_model is not None:
                # 使用本地模型生成文本
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            
                outputs = self.finetune_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    no_repeat_ngram_size=2,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            else:
                # 使用 API 生成文本
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=num_return_sequences,
                )

                generated_texts = [choice.message.content for choice in response.choices]

                # 记录 token 使用情况
                total_tokens = response.usage.total_tokens
                logger.info(f"Total tokens used: {total_tokens}")

            # 如果需要截断
            if truncate:
                generated_texts = [text[:max_tokens] for text in generated_texts]

        except Exception as e:
            logger.error(f"An error occurred during text generation: {str(e)}", exc_info=True)
            return None

        finally:
            logger.info(f"Generated text for prompt: {prompt[:50]}... (max_tokens: {max_tokens}, temperature: {temperature})")

        # 如果只请求一个序列，直接返回字符串而不是列表
        if num_return_sequences == 1 and generated_texts:
            return generated_texts[0]
        else:
            return generated_texts

    @error_handler  
    def translate_text(self, text: str, target_language: str = "zh") -> str:  
        if self.translation_pipeline is None:  
            self.translation_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ROMANCE", device=self.device)  
        translation = self.translation_pipeline(text, target_language=target_language)  
        translated_text = translation[0]['translation_text'] if translation else text  
        logger.info(f"Translated text from {text[:50]}... to {target_language}")  
        return translated_text  

    @error_handler  
    def classify_image(self, image: Union[str, Image.Image], top_k: int = 5) -> List[Dict[str, Any]]:  
        if self.image_classification_pipeline is None:  
            self.image_classification_pipeline = pipeline("image-classification", model="microsoft/resnet-50", device=self.device)  
        if isinstance(image, str):  
            image = Image.open(image)  
        results = self.image_classification_pipeline(image, top_k=top_k)  
        logger.info(f"Classified image with top {top_k} labels")  
        return results  
    
    @error_handler
    def fine_tune(self, train_texts, epochs=1, learning_rate=2e-5, batch_size=2):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.finetune_model.resize_token_embeddings(len(self.tokenizer))
    
        train_dataset = CustomDataset(train_texts, self.tokenizer, max_length=128)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(self.finetune_model.parameters(), lr=learning_rate)
        self.finetune_model.train()

        for epoch in range(epochs):
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.finetune_model(**inputs)
            
                loss = outputs.loss
                print(f"Calculated Loss: {loss.item()}")

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f"Epoch {epoch+1} completed.")
    
    @error_handler
    def save_model(self, path: str):
        self.finetune_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")
    
    @error_handler
    def load_model(self, path: str):
        self.finetune_model = AutoModelForCausalLM.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'  # 确保左侧填充
        logger.info(f"Model loaded from {path}")

    def use_api_model(self, model_name="gpt-3.5-turbo-0125"):
        self.finetune_model = None
        self.model_name = model_name
        logger.info(f"Switched to API model: {model_name}")
    
    @error_handler
    def generate_image_caption(self, image: Union[str, Image.Image]) -> str:
        if isinstance(image, str):
            image = Image.open(image)
        captions = self.image_captioning_pipeline(image)
        caption = captions[0]['generated_text'] if captions else ''
        logger.info("Generated image caption")
        return caption

    @error_handler
    def answer_question(self, context: str, question: str) -> str:
        qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=self.device)
        result = qa_pipeline(question=question, context=context)
        logger.info(f"Answered question: {question[:50]}...")
        return result['answer']

    @error_handler
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=self.device)
        result = sentiment_pipeline(text)[0]
        logger.info(f"Analyzed sentiment for text: {text[:50]}...")
        return {"label": result['label'], "score": result['score']}

    @error_handler
    def summarize_text(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=self.device)
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        logger.info(f"Generated summary for text: {text[:50]}...")
        return summary

    @error_handler
    def switch_model(self, model_name: str):
        if model_name == self.model_name:
            logger.info(f"Model {model_name} is already loaded")
            return
    
        if model_name.startswith("gpt"):
            # 对于 GPT 模型，我们只需要更新 model_name
            self.model_name = model_name
            logger.info(f"Switched to API model: {model_name}")
        else:
            # 对于本地模型，我们需要加载新的模型和分词器
            self.finetune_model, self.tokenizer = self._load_finetune_model_and_tokenizer(model_name)
            self.finetune_model_name = model_name
            logger.info(f"Switched to local model: {model_name}")

    def cleanup(self):  
        # 释放资源  
        del self.model  
        del self.tokenizer  
        del self.translation_pipeline  
        del self.image_classification_pipeline  
        del self.image_captioning_pipeline  
        torch.cuda.empty_cache()  
        logger.info("Resources cleaned up") 
        
# 主函数修改示例  
if __name__ == "__main__":  
    ai = GenerativeAI()  
    
    try:  
        # 文本生成示例  
        prompt = "The quick brown fox"  
        result = ai.generate_text(prompt)  
        print("Generated Text:", result)  

        # 文本翻译示例  
        text = "Hello, world!"  
        translation = ai.translate_text(text)  
        print("Translated Text:", translation) 
        
        # 图像分类示例
        image_path = "path/to/your/image.jpg"
        classification = ai.classify_image(image_path)
        print("Image Classification:", classification)

        # 问答示例
        context = "The capital of France is Paris. It is known for its beautiful architecture and cuisine."
        question = "What is the capital of France?"
        answer = ai.answer_question(context, question)
        print("Answer:", answer)

        # 情感分析示例
        sentiment_text = "I love this product! It's amazing!"
        sentiment = ai.analyze_sentiment(sentiment_text)
        print("Sentiment:", sentiment)

        # 文本摘要示例
        long_text = "Long text to be summarized..." * 10
        summary = ai.summarize_text(long_text)
        print("Summary:", summary)

        # 图像描述生成示例
        image_path = "path/to/your/image.jpg"
        caption = ai.generate_image_caption(image_path)
        print("Image Caption:", caption)

        # 微调示例
        train_data = ["Example text 1", "Example text 2", "Example text 3"]
        ai.fine_tune(train_data, epochs=1)  # 使用较少的 epoch 进行演示

        # ... (rest of the main function code)

    except Exception as e:  
        print(f"An error occurred: {str(e)}")  
    finally:  
        ai.cleanup()  
