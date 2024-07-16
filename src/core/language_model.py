import openai
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
import json
from datetime import datetime, timedelta

load_dotenv()

class CacheManager:
    def __init__(self, cache_file: str = "cache.json", expiration_days: int = 1):
        self.cache_file = cache_file
        self.expiration_days = expiration_days
        self.cache = self.load_cache()

    def load_cache(self) -> dict:
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            item = self.cache[key]
            if datetime.now() < datetime.fromisoformat(item['expiration']):
                return item['value']
            else:
                del self.cache[key]
                self.save_cache()
        return None

    def set(self, key: str, value: str):
        expiration = (datetime.now() + timedelta(days=self.expiration_days)).isoformat()
        self.cache[key] = {'value': value, 'expiration': expiration}
        self.save_cache()

class LanguageModel:
    def __init__(self, default_model: str = "gpt-3.5-turbo"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        self.default_model = default_model
        self.cache_manager = CacheManager()

    def generate_response(self, prompt: str, context: str = "", model: Optional[str] = None) -> str:
        model = model or self.default_model
        cache_key = f"{model}:{context}:{prompt}"
        
        # 尝试从缓存中获取响应
        cached_response = self.cache_manager.get(cache_key)
        if cached_response:
            return cached_response

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.choices[0].message.content

            # 将响应存入缓存
            self.cache_manager.set(cache_key, result)

            return result
        except Exception as e:
            print(f"Error in generating response: {e}")
            return "Sorry, I couldn't generate a response at this time."

    def get_available_models(self) -> List[str]:
        try:
            models = openai.Model.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"Error in fetching available models: {e}")
            return [self.default_model]

    def set_api_key(self, api_key: str):
        self.api_key = api_key
        openai.api_key = self.api_key

    def change_default_model(self, model: str):
        if model in self.get_available_models():
            self.default_model = model
        else:
            print(f"Model {model} is not available. Keeping the current default model.")

    def get_model_info(self, model: Optional[str] = None) -> Dict:
        model = model or self.default_model
        try:
            model_info = openai.Model.retrieve(model)
            return {
                "id": model_info.id,
                "created": model_info.created,
                "owned_by": model_info.owned_by,
            }
        except Exception as e:
            print(f"Error in fetching model info: {e}")
            return {}