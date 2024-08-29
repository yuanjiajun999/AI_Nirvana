import json  
import logging  
import os  
from typing import Any, Dict   
from src.api_config import APIConfig


class Config:  
    def __init__(self, config_file: str = "config.json"):  
        self.logger = logging.getLogger(__name__)  
        self.config_file = os.path.abspath(config_file)  
        self.config = self.load_config()  
        
        # 从配置文件获取API键和基础URL
        self.api_key: str = self.config.get('api_key', '')  
        self.api_base: str = self.config.get('api_base', "https://api.gptsapi.net/v1")  
        
        # 检查OPENAI_API_KEY，优先从config.json读取，否则从环境变量读取
        self.openai_api_key = self.config.get('api_key') or os.getenv("API_KEY")  
        if not self.openai_api_key:  
            raise ValueError("OPENAI_API_KEY not found in config file or environment variables")  
        
        # 模型名称，从配置文件或环境变量读取
        self.model: str = self.config.get('model', "gpt-3.5-turbo-0125")  
        
        if not self.validate_config():  
            self.logger.error("Configuration validation failed")  
            raise ValueError("Invalid configuration")  
        
        # 预定义的响应示例
        self.predefined_responses: Dict[str, str] = {  
            "introduce_yourself": "Hello, I am the AI assistant created for the AI Nirvana project. I'm here to help you with a variety of tasks.",  
            "how_are_you": "I'm doing well, thank you for asking.",  
            "what_can_you_do": "I can assist you with a wide range of tasks, such as answering questions, generating content, summarizing text, and performing sentiment analysis.",  
        }  
        
    def load_config(self) -> Dict[str, Any]:  
        default_config = {  
            "model": APIConfig.MODEL_NAME,  
            "log_level": "INFO",  
            "max_input_length": 1000,  
            "api_key": APIConfig.API_KEY,  
            "api_base": APIConfig.API_BASE,  
            "max_context_length": 5,  
        }  

        if not os.path.exists(self.config_file):  
            return self.create_default_config(default_config)  

        try:  
            with open(self.config_file, "r") as f:  
                loaded_config = json.load(f)  
            
            config = default_config.copy()  
            config.update(loaded_config)  
            
            return self.update_config_from_env(config)  
        except json.JSONDecodeError:  
            self.logger.error(f"Error decoding {self.config_file}. Creating default config.")  
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
        print(f"Validating configuration...")  
        print(f"API Key: {'*' * (len(self.openai_api_key) - 4) + self.openai_api_key[-4:] if self.openai_api_key else 'Not set'}")  
        print(f"API Base: {self.api_base}")  
        print(f"Model: {self.config.get('model', 'Not set')}")  

        required_keys = ["model", "log_level", "max_input_length", "api_base"]  
        for key in required_keys:  
            if key not in self.config:  
                self.logger.error(f"Missing required configuration key: {key}")  
                print(f"Validation failed: Missing {key}")  
                return False  

        if not self.openai_api_key:  
            self.logger.error("OPENAI_API_KEY is missing or empty")  
            print("Validation failed: OPENAI_API_KEY is missing or empty")  
            return False  

        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]  
        if self.config.get("log_level") not in valid_log_levels:  
            self.logger.error(f"Invalid log level: {self.config.get('log_level')}")  
            print(f"Validation failed: Invalid log level {self.config.get('log_level')}")  
            return False  

        print("Configuration validation successful")  
        return True  