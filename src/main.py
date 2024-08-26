import argparse
import sys
import time
import atexit
import psutil
from retry import retry
from src.config import Config
from src.commands import AINirvana, handle_command
from src.command_data import AVAILABLE_COMMANDS
from src.help_info import print_help
from flask import Flask, jsonify, request
import logging

logger = logging.getLogger(__name__)

# 清理函数
def cleanup() -> None:
    logger.info("执行清理操作...")
    # 在这里添加任何必要的清理代码，例如关闭数据库连接，释放资源等

atexit.register(cleanup)

@retry(exceptions=(Exception,), tries=3, delay=5, backoff=2, logger=logger)
def initialize_system(config: Config) -> AINirvana:
    logger.info("Initializing AI Nirvana system...")
    ai_nirvana = AINirvana(config)
    logger.info("AI Nirvana system initialized successfully.")
    return ai_nirvana

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

                response = self.ai_nirvana.process(input_text)
                return jsonify({"response": response})
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                return jsonify({"error": "An internal error occurred"}), 500

    def run(self):
        logger.info("Starting Flask server...")
        self.app.run(host="0.0.0.0", port=8000, use_reloader=False)

def run_server(config: Config):
    server = AIServer(config)
    server.run()

def main(config: Config, mode: str) -> None:
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1} to start the system")
            
            if mode == "cli":
                ai_nirvana = initialize_system(config)
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
                            break  # 使用 break 来退出循环

                    except KeyboardInterrupt:
                        print("\n程序被用户中断。")
                        break  # 使用 break 来退出循环
                    except Exception as e:
                        logger.error(f"处理输入时发生错误: {str(e)}")
                        print("抱歉，处理您的输入时遇到了问题。请重试或输入其他命令。")

                # CLI 模式结束后，不再继续执行
                return

            elif mode == "server":
                run_server(config)
            else:
                logger.error(f"Invalid mode: {mode}")
                return

        except Exception as e:
            logger.error(f"Error during startup (Attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Max retries reached. Unable to start the system.")
                raise

    print("感谢使用 AI Nirvana 智能助手，再见！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Nirvana Assistant")
    parser.add_argument("--config", default="config.json", help="Path to the configuration file")
    parser.add_argument("--mode", choices=["cli", "server"], default="cli", help="Run in CLI or server mode")
    args = parser.parse_args()

    try:
        config = Config(args.config)
        if not config.validate_config():
            logger.error("Configuration validation failed. Please check your config file.")
            sys.exit(1)

        main(config, args.mode)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
    finally:
        # 确保在程序退出时调用清理函数
        cleanup()
        # 终止当前进程及其子进程
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            child.terminate()
        current_process.terminate()

    sys.exit(0)