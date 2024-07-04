import os
import requests
import json

# 设置环境变量
API_KEY = "sk-vRu126d626325944f7040b39845200bafd41123d8f3g48Ol"  # 请确保这是正确的 API 密钥
API_BASE = "https://api.gptsapi.net/v1"

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
    except Exception as e:
        print(f"API 调用错误: {e}")
        print(f"请求头: {headers}")
        print(f"请求数据: {json.dumps(data, ensure_ascii=False)}")
        if hasattr(e, 'response'):
            print(f"响应状态码: {e.response.status_code}")
            print(f"响应内容: {e.response.text}")
        return "Error occurred during API call"

def process_input(input_text):
    prompt = f"Provide a helpful and insightful summary of the following in one or two sentences, adding relevant context or implications where appropriate: {input_text}"
    return chat_completion(prompt)

# 测试用例
test_cases = [
    "LangChain is a framework for developing applications powered by language models.",
    "Python是一种高级编程语言。",
    "机器学习是人工智能的一个子集。",
    "气候变化正在影响全球天气模式。",
    "物联网（IoT）将日常设备连接到互联网。",
    "量子计算使用量子力学现象来执行数据运算。"
]

if __name__ == "__main__":
    for test in test_cases:
        print(f"\n输入: {test}")
        try:
            output = process_input(test)
            print(f"输出: {output}")
        except Exception as e:
            print(f"处理 '{test}' 时出错: {e}")
