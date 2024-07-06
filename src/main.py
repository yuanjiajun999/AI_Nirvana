import os
import requests
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 从环境变量获取 API 密钥和基础 URL
API_KEY = os.getenv('API_KEY')
API_BASE = os.getenv('API_BASE')

def chat_completion(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.RequestException as e:
        print(f"API 调用错误: {e}")
        return "Error occurred during API call"

def process_input(input_text):
    prompt = f"Summarize the following in one or two sentences: {input_text}"
    return chat_completion(prompt)

def main():
    while True:
        user_input = input("请输入要总结的文本（输入 'quit' 退出）: ")
        if user_input.lower() == 'quit':
            break
        summary = process_input(user_input)
        print(f"总结: {summary}\n")

if __name__ == "__main__":
    main()
