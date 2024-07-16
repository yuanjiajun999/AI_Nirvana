import time
from functools import lru_cache

import requests

from .model_interface import ModelInterface


class APIModel(ModelInterface):
    def __init__(self, api_key, api_url, max_retries=3, retry_delay=1):
        self.api_key = api_key
        self.api_url = api_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @lru_cache(maxsize=128)
    def generate(self, prompt):
        return self._make_api_call(prompt)

    def _make_api_call(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "prompt": prompt,
            "max_tokens": 150
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=data, timeout=10)
                response.raise_for_status()
                return response.json()['choices'][0]['text'].strip()
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    return f"Error: Unable to get response after {self.max_retries} attempts. Last error: {str(e)}"
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff

    def summarize(self, text):
        return self._make_api_call(f"Summarize the following text: {text}")