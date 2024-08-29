import argparse  
import os  
import json 
import sys  
import time  
import atexit  
import psutil  
from src.config import Config  
from src.core.api_client import ApiClient  
from src.commands import AINirvana, handle_command  
from src.command_data import AVAILABLE_COMMANDS  
from src.help_info import get_help  
from flask import Flask, jsonify, request  
import logging  
from src.core.model_factory import ModelFactory  
from src.core.language_model import LanguageModel  

# 设置日志  
logger = logging.getLogger(__name__)  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')  

# 读取 config.json 文件
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# 从配置中获取 API 设置，用于调试
api_key = config.get("api_key")
api_base = config.get("api_base")
model_name = config.get("model")
if api_key:  
    print(f"API Key loaded: {api_key[:5]}...{api_key[-5:]}")  
else:  
    print("API Key not found in environment variables")  

print(f"API Base URL: {api_base}")  
print(f"Model Name: {model_name}")  

def cleanup() -> None:  
    logger.info("执行清理操作...")  
    # 在这里添加任何必要的清理代码  

atexit.register(cleanup)  

def initialize_system(config: Config) -> AINirvana:  
    logger.info("Initializing AI Nirvana system...")  
    try:  
        # 注册 LanguageModel  
        ModelFactory.register_model("LanguageModel", LanguageModel)  
        
        # 创建 ApiClient 实例，只传递 config 对象
        api_client = ApiClient(config)  
        
        # 创建 AINirvana 实例，传入 config 和 api_client  
        logger.info(f"Creating AINirvana with config: {config}, api_client: {api_client}")  
        ai_nirvana = AINirvana(config, api_client)  
        
        logger.info("AI Nirvana system initialized successfully.")  
        return ai_nirvana  
    except Exception as e:  
        logger.error(f"Error during AI Nirvana initialization: {str(e)}")  
        raise

    
class AIServer:  
    def __init__(self, config: Config):  
        self.ai_nirvana = initialize_system(config)  
        self.app = Flask(__name__)  
        self.setup_routes()  

    def setup_routes(self):  
        @self.app.route("/process", methods=["POST"])  
        def process():  
            try:  
                input_text = request.json.get("input")  
                if not input_text:  
                    return jsonify({"error": "No input provided"}), 400  

                result = handle_command(input_text, self.ai_nirvana)  
                return jsonify(result)  
            except Exception as e:  
                logger.error(f"Error processing request: {str(e)}")  
                return jsonify({"error": "An internal error occurred"}), 500  

    def run(self):  
        logger.info("Starting Flask server...")  
        self.app.run(host="0.0.0.0", port=8000, use_reloader=False)  

def run_server(config: Config):  
    server = AIServer(config)  
    server.run()  

def main(config_file: str, mode: str) -> None:  
    try:  
        config = Config(config_file)  
        ai_nirvana = initialize_system(config)  
        
        if mode == "cli":  
            print("欢迎使用 AI Nirvana 智能助手！")  
            print("输入 'help' 查看可用命令。")  

            while True:  
                try:  
                    user_input = input("\n请输入您的问题或命令（输入空行发送）：").strip()  
                    if not user_input:  
                        print("您似乎没有输入任何内容。请输入一些文字或命令。")  
                        continue  

                    logger.info(f"User input: {user_input}")  
                    print("正在处理命令...")  
                    result = handle_command(user_input, ai_nirvana)  

                    if not result.get("continue", True):  
                        print(result.get("message", "再见！"))  
                        break  

                except KeyboardInterrupt:  
                    print("\n程序被用户中断。")  
                    break  
                except Exception as e:  
                    logger.error(f"处理输入时发生错误: {str(e)}")  
                    print("抱歉，处理您的输入时遇到了问题。请重试或输入其他命令。")  

        elif mode == "server":  
            run_server(config)  
        else:  
            logger.error(f"Invalid mode: {mode}")  
            print("错误：无效的模式。请选择 'cli' 或 'server'。")  

    except Exception as e:  
        logger.error(f"Error during startup: {str(e)}")  
        raise  

    print("感谢使用 AI Nirvana 智能助手，再见！")  

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="AI Nirvana Assistant")  
    parser.add_argument("--config", default="config.json", help="Path to the configuration file")  
    parser.add_argument("--mode", choices=["cli", "server"], default="cli", help="Run in CLI or server mode")  
    args = parser.parse_args()  

    try:  
        main(args.config, args.mode)  
    except Exception as e:  
        logger.error(f"An unexpected error occurred: {str(e)}")  
    finally:  
        cleanup()  

    sys.exit(0)
