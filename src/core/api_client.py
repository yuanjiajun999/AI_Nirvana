
import openai  
from src.config import Config  
import logging  

# 设置日志记录  
logger = logging.getLogger(__name__)  
print("api_client.py is being imported")  

class ApiClient:  
    def __init__(self, config: Config):  
        logger.info("Initializing ApiClient")  
        try:  
            self.config = config  
            if not isinstance(config.api_key, str) or not config.api_key:  
                raise ValueError("API key must be a non-empty string")  
            self.api_key = config.api_key  
            self.api_base = config.api_base  
            self.model = config.model  
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)  
            logger.info(f"ApiClient initialized with base URL: {self.api_base} and model: {self.model}")  
        except AttributeError as e:  
            logger.error(f"Configuration error: {str(e)}")  
            raise ValueError("Invalid configuration. Please check your config object.")  
        except Exception as e:  
            logger.error(f"Unexpected error during ApiClient initialization: {str(e)}")  
            raise

    def chat_completion(self, messages):  
        logger.info(f"Sending chat completion request with {len(messages)} messages")  
        try:  
            response = self.client.chat.completions.create(  
                model=self.model,  
                messages=messages  
            )  
            return response  
        except openai.OpenAIError as e:  
            logger.error(f"Chat completion request failed: {str(e)}")  
            raise  
        except Exception as e:  
            logger.error(f"Unexpected error in chat completion: {str(e)}")  
            raise  

    def test_connection(self):  
        logger.info("Testing API connection")  
        try:  
            response = self.chat_completion([{"role": "user", "content": "Hello"}])  
            if response.choices and response.choices[0].message.content:  
                logger.info("API connection test successful")  
                return True  
            logger.warning("API connection test: Unexpected response format")  
            return False  
        except Exception as e:  
            logger.error(f"API connection test failed: {str(e)}")  
            raise ConnectionError(f"Failed to connect to API: {str(e)}")
        
# 可选：添加一个简单的使用示例
if __name__ == "__main__":
    # 注意：这里假设 Config 类在实际使用时会正确初始化
    config = Config()  # 这里应该使用实际的配置初始化方法
    client = ApiClient(config)
    try:
        if client.test_connection():
            print("Connection test passed")
        else:
            print("Connection test failed")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
