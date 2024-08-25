# api_client.py  

import requests  

class ApiClient:  
    def __init__(self, api_key):  
        self.api_key = api_key  
        self.base_url = "https://api.gptsapi.net/v1/chat/completions"  # 使用正确的 API 端点  

    def chat_completion(self, messages):  
        headers = {  
            "Authorization": f"Bearer {self.api_key}",  
            "Content-Type": "application/json"  
        }  
        data = {  
            "model": "gpt-3.5-turbo",  # 或您使用的其他模型  
            "messages": messages  
        }  
        response = requests.post(self.base_url, json=data, headers=headers)  
        response.raise_for_status()  # 这会在 HTTP 错误时抛出异常  
        return response.json()