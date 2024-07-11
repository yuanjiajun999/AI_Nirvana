import logging
from src.config import Config

def setup_logging(config: Config):
    log_level = config.get('log_level')
    logging.basicConfig(
        filename='ai_assistant.log',
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def handle_exception(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            return "I'm sorry, an error occurred. Please try again later."
    return wrapper

@handle_exception
def generate_response(prompt):
    # 核心响应生成逻辑
    return response