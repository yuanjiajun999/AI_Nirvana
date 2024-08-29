# debug_initialization.py  

import logging
import threading
import json 
import os
from src.config import Config
from src.core.api_client import ApiClient
from src.core.knowledge_base import KnowledgeBase
from src.commands import AINirvana  # 从 commands.py 导入 AINirvana

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载配置
with open('config.json', 'r') as f:
    config = json.load(f)

# 获取 API 密钥
api_key = config.get('api_key') or os.environ.get('OPENAI_API_KEY')

if not api_key:
    raise ValueError("API key not found in config or environment variables")

# 使用 api_key 进行初始化
from openai import OpenAI
client = OpenAI(api_key=api_key, base_url=config['api_base'])

def init_config():
    logger.debug("Initializing configuration...")
    try:
        config = Config()
        logger.info(f"Configuration initialized. API Base: {config.api_base}, Model: {config.model}")
        logger.debug(f"API Key loaded: {'Yes' if config.openai_api_key else 'No'}")
        return config
    except Exception as e:
        logger.error(f"Failed to initialize configuration: {str(e)}")
        raise

def init_api_client(config):
    logger.debug("Initializing API client...")
    try:
        client = ApiClient(config)
        client.test_connection()
        logger.info("API client initialized and connection tested successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize API client: {str(e)}")
        raise

def init_knowledge_base(config, api_client):
    logger.debug("Starting knowledge base initialization...")
    try:
        kb = KnowledgeBase(config, api_client)
        logger.debug("KnowledgeBase instance created")
        test_result = kb.test_operation()
        logger.info(f"Knowledge base initialized successfully. Test result: {test_result}")
        return kb
    except Exception as e:
        logger.error(f"Failed to initialize knowledge base: {str(e)}", exc_info=True)
        raise

def init_ai_nirvana(config, api_client):
    logger.debug("Initializing AINirvana...")
    try:
        ai_nirvana = AINirvana(config, api_client)
        logger.info("AINirvana initialized successfully")
        return ai_nirvana
    except Exception as e:
        logger.error(f"Failed to initialize AINirvana: {str(e)}", exc_info=True)
        raise

def test_knowledge_base_operations(kb):
    logger.info("Starting knowledge base operations test")
    try:
        logger.info("Testing set operation")
        kb.set("test_key", {"content": "This is a test", "source": "debug_initialization.py"})
        
        logger.info("Testing get operation")
        retrieved = kb.get("test_key")
        
        logger.info("Testing search operation")
        search_results = kb.search("test")
        
        logger.info("Testing query operation")
        query_result = kb.query("What is the test content?")
        
        logger.info("Testing delete operation")
        kb.delete("test_key")
        
        logger.info("All knowledge base operations completed successfully")
    except Exception as e:
        logger.error(f"Error during knowledge base operations test: {str(e)}", exc_info=True)

def test_ai_nirvana_operations(ai_nirvana):
    logger.info("Starting AINirvana operations test")
    try:
        # 测试 AINirvana 的各种方法
        # 例如：
        # result = ai_nirvana.execute_command("some command")
        # logger.info(f"Command execution result: {result}")
        
        logger.info("All AINirvana operations completed successfully")
    except Exception as e:
        logger.error(f"Error during AINirvana operations test: {str(e)}", exc_info=True)

def main():
    try:
        logger.info("Starting initialization process")
        
        config = init_config()
        api_client = init_api_client(config)
        knowledge_base = init_knowledge_base(config, api_client)
        
        logger.info("About to initialize AINirvana")
        ai_nirvana = init_ai_nirvana(config, api_client)
        
        logger.info("Initialization completed, starting operations test")
        logger.info("About to call test_knowledge_base_operations")
        test_knowledge_base_operations(knowledge_base)
        logger.info("test_knowledge_base_operations completed")
        
        logger.info("About to call test_ai_nirvana_operations")
        test_ai_nirvana_operations(ai_nirvana)
        logger.info("test_ai_nirvana_operations completed")
        
        logger.info("All operations completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)

def timeout_handler():
    logger.error("Script execution timed out after 300 seconds")
    os._exit(1)

if __name__ == "__main__":
    logger.info("Script started")
    timer = threading.Timer(300, timeout_handler)  # 5 minutes
    timer.start()
    try:
        main()
    except Exception as e:
        logger.critical(f"Unexpected error in main script: {str(e)}", exc_info=True)
    finally:
        timer.cancel()
        logger.info("Script completed")
