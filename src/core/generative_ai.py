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
        return {key: val.squeeze(0) for key, val in encoding.items()}

class GenerativeAI:  
    def __init__(self):  
        self.client = OpenAI(  
            api_key=os.getenv("API_KEY"),  
            base_url=os.getenv("API_BASE")  
        )  
        self.model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo-0125")   

    @error_handler  
    def _load_model_and_tokenizer(self):  
        model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)  
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)  
        return model, tokenizer  

    def generate_text(self, prompt, max_tokens=1000, temperature=0.7, num_return_sequences=1, truncate=False):  
        try:  
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

            # 如果只请求一个序列，直接返回字符串而不是列表  
            if num_return_sequences == 1:  
                return generated_texts[0]  
            else:  
                return generated_texts  

        except Exception as e:  
            logger.error(f"An error occurred during text generation: {str(e)}", exc_info=True)  
            return None  

        finally:  
            # 添加详细的日志记录  
            logger.info(f"Generated text for prompt: {prompt[:50]}... (max_tokens: {max_tokens}, temperature: {temperature})")  
 
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
    def fine_tune(self, train_data: List[str], epochs: int = 3, learning_rate: float = 2e-5, batch_size: int = 4):
        train_texts, val_texts = train_test_split(train_data, test_size=0.1)
        
        train_dataset = CustomDataset(train_texts, self.tokenizer, max_length=128)
        val_dataset = CustomDataset(val_texts, self.tokenizer, max_length=128)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                total_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Average loss: {avg_loss:.4f}")
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Validation loss: {avg_val_loss:.4f}")
        
        logger.info("Fine-tuning completed.")

    @error_handler
    def save_model(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")

    @error_handler
    def load_model(self, path: str):
        self.model = AutoModelForCausalLM.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        logger.info(f"Model loaded from {path}")

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