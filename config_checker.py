import os
import json
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_file='config.json'):
    """
    加载配置文件
    """
    try:
        with open(config_file, 'r') as file:
            config = json.load(file)
            logger.info(f"配置文件 {config_file} 加载成功")
            return config
    except Exception as e:
        logger.error(f"加载配置文件时出错: {e}")
        return None

def check_environment_variables(config):
    """
    检查配置文件中的关键配置项，并与环境变量对比
    """
    issues_found = False
    
    # 检查 API Key
    api_key = config.get('api_key')
    env_api_key = os.getenv('API_KEY')
    if not api_key:
        logger.error("配置文件中缺少 'api_key'")
        issues_found = True
    else:
        logger.info("配置文件中包含 'api_key'")

    if not env_api_key:
        logger.error("环境变量中缺少 'API_KEY'")
        issues_found = True
    else:
        logger.info("环境变量中包含 'API_KEY'")
    
    # 检查 Base URL
    api_base = config.get('api_base')
    env_api_base = os.getenv('API_BASE')
    if not api_base:
        logger.error("配置文件中缺少 'api_base'")
        issues_found = True
    else:
        logger.info("配置文件中包含 'api_base'")

    if not env_api_base:
        logger.error("环境变量中缺少 'API_BASE'")
        issues_found = True
    else:
        logger.info("环境变量中包含 'API_BASE'")

    if issues_found:
        logger.error("检测到问题，请检查配置文件和环境变量的设置")
    else:
        logger.info("配置文件和环境变量均正常")

if __name__ == "__main__":
    config = load_config()

    if config:
        check_environment_variables(config)
    else:
        logger.error("无法加载配置文件，终止检查")
