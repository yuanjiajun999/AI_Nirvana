# api_client.py

import requests
import logging

# 设置日志记录
logger = logging.getLogger(__name__)

class ApiClient:
    def __init__(self, api_key, api_base=None):
        self.api_key = api_key
        self.api_base = api_base or "https://api.gptsapi.net/v1"
        self.chat_completion_url = f"{self.api_base}/chat/completions"

    def chat_completion(self, messages):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-3.5-turbo",  # 或您使用的其他模型
            "messages": messages
        }
        try:
            response = requests.post(self.chat_completion_url, json=data, headers=headers)
            response.raise_for_status()  # 这会在 HTTP 错误时抛出异常
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Chat completion request failed: {str(e)}")
            raise

    def test_connection(self):
        try:
            # 使用 chat completion 端点来测试连接
            test_message = [{"role": "user", "content": "Hello, API!"}]
            response = self.chat_completion(test_message)
            logger.info("API connection test successful")
            return True
        except requests.RequestException as e:
            logger.error(f"API connection test failed: {str(e)}")
            return False