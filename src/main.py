<<<<<<< HEAD
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from src.core.ai_assistant import AIAssistant
from src.core.local_model import LocalModel
from src.core.api_model import APIModel
from src.config import Config, predefined_responses
from src.dialogue_manager import DialogueManager
from src.ui import print_user_input, print_assistant_response, print_dialogue_context

def setup_logger(log_level):
    logging.basicConfig(filename='ai_assistant.log', level=log_level, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

def get_model(config):
    model_name = config.get('model')
    if model_name == 'local':
        return LocalModel()
    elif model_name == 'api':
        api_key = config.get('api_key')
        api_url = config.get('api_url')
        if not api_key or not api_url:
            raise ValueError("API key and URL must be set in config for API model")
        return APIModel(api_key, api_url)
    raise ValueError(f"Unknown model: {model_name}")

def cooperative_algorithm(user_input):
    if user_input.lower() in predefined_responses:
        return predefined_responses[user_input.lower()]
    else:
        return "I'm afraid I don't have a specific response for that query. How else can I assist you?"

def handle_user_input(user_input):
    if user_input.lower() in predefined_responses:
        return predefined_responses[user_input.lower()]
    else:
        # Use the cooperative algorithm to generate a response
        response = cooperative_algorithm(user_input)
        return response

def main():
    config = Config()
    setup_logger(config.get('log_level'))
    
    model = get_model(config)
    assistant = AIAssistant(model)
    dialogue_manager = DialogueManager(max_history=5)
    
    print("欢迎使用 AI Nirvana 智能助手！")
    print("输入 'quit' 退出程序。")
    print("您可以询问问题、请求生成内容，或者输入长文本进行摘要。")
    
    max_input_length = config.get('max_input_length')
    
    while True:
        user_input = input("\n请输入您的问题或文本（或输入 'quit' 退出）：\n")
        
        if user_input.lower() == 'quit':
            print("谢谢使用,再见!")
            break
        
        if len(user_input) > max_input_length:
            response = assistant.summarize(user_input)
            print_user_input(user_input)
            print("\n摘要：")
            print_assistant_response(response)
        else:
            response = assistant.generate_response(user_input)
            print_user_input(user_input)
            print("\n回答：")
            print_assistant_response(response)

        dialogue_manager.add_to_history(user_input, response)
        print_dialogue_context(dialogue_manager.get_dialogue_context())

if __name__ == "__main__":
    main()
=======
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
>>>>>>> df302b19101ed1290fe7bf533dd80398212ef881
