import json
import logging
import os
from typing import Any, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

class APIConfig:
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo-0125")
    API_KEY = os.getenv("API_KEY")
    API_BASE = os.getenv("API_BASE", "https://api.gptsapi.net/v1")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))

class Config:
    def __init__(self, config_file: str = "config.json"):
        self.logger = logging.getLogger(__name__)
        self.config_file = config_file  
        self.config = self.load_config()  
        self.api_key = self.config.get('api_key')
        if not self.validate_config():  
            logging.error("Configuration validation failed")  
            raise ValueError("Invalid configuration") 
        self.predefined_responses = {
            "introduce_yourself": "Hello, I am the AI assistant created for the AI Nirvana project. I'm here to help you with a variety of tasks.",
            "how_are_you": "I'm doing well, thank you for asking.",
            "what_can_you_do": "I can assist you with a wide range of tasks, such as answering questions, generating content, summarizing text, and performing sentiment analysis.",
        }

    def load_config(self) -> Dict[str, Any]:
        """加载配置文件，如果不存在则创建默认配置"""
        default_config = {
            "model": "gpt-3.5-turbo",
            "log_level": "INFO",
            "max_input_length": 1000,
            "api_key": "",
            "max_context_length": 5,
            # ... 其他默认值 ...
        }

        if not os.path.exists(self.config_file):
            return self.create_default_config(default_config)

        try:
            with open(self.config_file, "r") as f:
                loaded_config = json.load(f)
        
            # 使用默认配置，然后用加载的配置更新它
            config = default_config.copy()
            config.update(loaded_config)
        
            return self.update_config_from_env(config)
        except json.JSONDecodeError:
            logging.error(
                f"Error decoding {self.config_file}. Creating default config."
            )
            return self.create_default_config(default_config)

    def create_default_config(self, default_config: Dict[str, Any]) -> Dict[str, Any]:
        """创建默认配置并保存到文件"""
        with open(self.config_file, "w") as f:
            json.dump(default_config, f, indent=4)
        logging.info(f"Created default config file: {self.config_file}")
        return default_config

    def update_config_from_env(self, config: Dict[str, Any]) -> Dict[str, Any]:  
        """从环境变量更新配置"""  
        env_mapping = {  
            "OPENAI_API_KEY": "api_key",  
            "MODEL_NAME": "model",  
            "LOG_LEVEL": "log_level",  
            "USE_GPU": "use_gpu",  
            "SYSTEM_PROMPT": "system_prompt",  
            "MAX_CONTEXT_LENGTH": "max_context_length",  
        }  
        for env_var, config_key in env_mapping.items():  
            env_value = os.getenv(env_var)  
            if env_value is not None:  
                if config_key == "api_key" and not env_value.strip():  
                    logging.warning("API key from environment variable is empty")  
                else:  
                    config[config_key] = env_value  
        return config

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """设置配置值并保存"""
        self.config[key] = value
        self.save_config()

    def save_config(self) -> None:
        """保存配置到文件"""
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=4)

    def get_predefined_response(self, key: str) -> str:
        """获取预定义响应"""
        return self.predefined_responses.get(
            key, "I'm sorry, I don't have a predefined response for that."
        )

    def validate_config(self) -> bool:  
        self.logger.info("Validating configuration...")
        """验证配置的完整性和正确性"""  
        print(f"Validating configuration...")  
        print(f"API Key: {'*' * (len(self.api_key) - 4) + self.api_key[-4:] if self.api_key else 'Not set'}")  
        print(f"API Base: {self.config.get('api_base', 'Not set')}")  
        print(f"Model: {self.config.get('model', 'Not set')}")  

        required_keys = [  
            "model",  
            "log_level",  
            "max_input_length",  
            "api_key",  
            "api_base",  
        ]  
        for key in required_keys:  
            if key not in self.config:  
                logging.error(f"Missing required configuration key: {key}")  
                print(f"Validation failed: Missing {key}")  
                return False  

        # 特别检查 API 密钥  
        if not self.config.get("api_key"):  
            logging.error("API key is missing or empty")  
            print("Validation failed: API key is missing or empty")  
            return False  

        # 添加其他验证逻辑，例如检查 API 密钥格式、日志级别是否有效等  
        # 例如：  
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]  
        if self.config.get("log_level") not in valid_log_levels:  
            logging.error(f"Invalid log level: {self.config.get('log_level')}")  
            print(f"Validation failed: Invalid log level {self.config.get('log_level')}")  
            return False  

        print("Configuration validation successful")  
        return True