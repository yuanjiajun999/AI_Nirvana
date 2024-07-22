import requests
import sys
import os

print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

api_key = "sk-vRu126d626325944f7040b39845200bafd41123d8f3g48Ol"
print(f"Using API Key: {api_key}")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": "gpt-3.5-turbo-0125",
    "messages": [{"role": "user", "content": "Hello, World!"}]
}

try:
    response = requests.post("https://api.gptsapi.net/v1/chat/completions", headers=headers, json=data)
    print(f"Response status code: {response.status_code}")
    print(f"Response headers: {response.headers}")
    print(f"Response content: {response.text}")
    response.raise_for_status()
    print("API Response:", response.json())
except requests.exceptions.RequestException as e:
    print("Error occurred:", str(e))
    if hasattr(e, 'response') and e.response is not None:
        print("Response content:", e.response.text)