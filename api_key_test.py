
import os
import sys

class APIClient:
    def __init__(self, api_key):
        self.api_key = api_key

    def test_connection(self):
        print(f"Testing connection with API key: {self.api_key[:5]}...{self.api_key[-5:]}")
        # 这里可以添加实际的API连接测试逻辑

def load_config():
    # 模拟从环境变量或配置文件加载API密钥
    return os.getenv('API_KEY', 'default_api_key')

def main():
    try:
        # 加载配置
        api_key = load_config()
        print(f"Loaded API key: {api_key[:5]}...{api_key[-5:]}")

        # 创建API客户端
        client = APIClient(api_key)

        # 测试连接
        client.test_connection()

        print("API key test completed successfully.")
    except AttributeError as e:
        print(f"AttributeError: {e}")
        print("This error suggests that the API key is being treated as an object instead of a string.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
