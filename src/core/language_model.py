import os  
from typing import Any, Dict, List, Optional  
import json  
from src.core.model_interface import ModelInterface  
from dotenv import load_dotenv  
from openai import OpenAI  

from src.utils.error_handler import ModelError, error_handler, logger  

load_dotenv()  

class LanguageModel(ModelInterface):  
    def __init__(self, config, api_client, model_name: str = "gpt-3.5-turbo-0125"):  
        self.config = config
        self.api_client = api_client
        self.model_name = model_name  
        self.api_key = config.api_key  # 从 config 获取 API key
        self.client = OpenAI(api_key=self.api_key, base_url=config.api_base)  
        logger.info(f"LanguageModel initialized with model: {self.model_name}") 

    def switch_model(self, new_model_name: str):
        self.model_name = new_model_name
        logger.info(f"LanguageModel switched to model: {self.model_name}")
        
    @error_handler  
    def generate_response(self, prompt: str, model: Optional[str] = None, context: Optional[str] = None) -> str:  
        model = model or self.model_name  
        messages = [{"role": "system", "content": context or "You are a helpful assistant."},  
                    {"role": "user", "content": prompt}]  
        
        try:  
            response = self.client.chat.completions.create(  
                model=model,  
                messages=messages  
            )  
            return response.choices[0].message.content  
        except Exception as e:  
            raise ModelError(f"Error generating response: {str(e)}")  

    @error_handler  
    def get_available_models(self) -> List[str]:  
        try:  
            models = self.client.models.list()  
            return [model.id for model in models.data]  
        except Exception as e:  
            logger.error(f"Error retrieving available models: {str(e)}")  
            return []  

    @error_handler  
    def change_model(self, new_model: str) -> None:  
        if new_model in self.get_available_models():  
            self.model_name = new_model  
            logger.info(f"Model changed to: {self.model_name}")  
        else:  
            raise ValueError(f"Model {new_model} is not available.")  

    @error_handler  
    def get_model_info(self) -> Dict[str, Any]:  
        try:  
            model_info = self.client.models.retrieve(self.model_name)  
            logger.info(f"Retrieved info for model: {self.model_name}")  
            return {  
                "id": model_info.id,  
                "created": model_info.created,  
                "owned_by": model_info.owned_by,  
            }  
        except Exception as e:  
            logger.error(f"Error in fetching model info: {e}")  
            raise ModelError(f"Failed to fetch model info: {str(e)}")  

    @error_handler  
    def analyze_sentiment(self, text: str) -> Dict[str, float]:  
        prompt = f"Analyze the sentiment of the following text and return a JSON object with keys 'positive', 'neutral', and 'negative', where the values are floats representing the probability of each sentiment:\n\n{text}"  
        try:  
            response = self.generate_response(prompt)  
            sentiment = json.loads(response)  
            logger.info(f"Sentiment analysis completed for text: {text[:50]}...")  
            return sentiment  
        except json.JSONDecodeError:  
            logger.error("Failed to parse sentiment analysis result")  
            raise ModelError("Failed to parse sentiment analysis result")  
        except Exception as e:  
            logger.error(f"Error in sentiment analysis: {e}")  
            raise ModelError(f"Failed to analyze sentiment: {str(e)}")  

    @error_handler  
    def summarize(self, text: str, max_length: int = 100) -> str:  
        prompt = f"请用中文简洁地总结以下文本，不超过{max_length}字：\n\n{text}"  
        try:  
            summary = self.generate_response(prompt)  
            logger.info(f"Summary generated for text: {text[:50]}...")  
            return summary  
        except Exception as e:  
            logger.error(f"Error in generating summary: {e}")  
            raise ModelError(f"Failed to generate summary: {str(e)}")  

    @error_handler  
    def translate(self, text: str, target_language: str) -> str:  
        prompt = f"Translate the following text to {target_language}:\n\n{text}"  
        try:  
            translated_text = self.generate_response(prompt)  
            logger.info(f"Text translated to {target_language}")  
            return translated_text  
        except Exception as e:  
            logger.error(f"Error in translation: {e}")  
            raise ModelError(f"Failed to translate text: {str(e)}")  

    def clear_context(self) -> None:  
        # 如果需要清除上下文，可以在这里实现相应的逻辑  
        logger.info("Context cleared")  

    # 实现 ModelInterface 中的其他抽象方法  
    def load_model(self, model_path: str) -> None:  
        # OpenAI API 不需要本地加载模型，所以这里可以是空实现  
        pass  

    def predict(self, input_data: Any) -> Any:  
        # 可以使用 generate_response 方法来实现预测  
        return self.generate_response(input_data)  

    def train(self, training_data: Any, labels: Any) -> None:  
        # OpenAI API 不支持直接训练，所以这里可以是空实现  
        pass  

    def evaluate(self, test_data: Any, test_labels: Any) -> Dict[str, float]:  
        # OpenAI API 不支持直接评估，所以这里可以是空实现  
        return {}  

    def save_model(self, model_path: str) -> None:  
        # OpenAI API 不需要本地保存模型，所以这里可以是空实现  
        pass  

    def preprocess_data(self, raw_data: Any) -> Any:  
        # 可以根据需要实现数据预处理逻辑  
        return raw_data  

    def fine_tune(self, fine_tuning_data: Any, fine_tuning_labels: Any) -> None:  
        # OpenAI API 不支持直接微调，所以这里可以是空实现  
        pass  

    def explain_prediction(self, input_data: Any, prediction: Any) -> str:  
        # 可以实现一个简单的解释逻辑  
        return f"Prediction '{prediction}' was made based on input: {input_data}"