import os
import sys
import argparse
from typing import Dict, Any
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.core.ai_assistant import AIAssistant
from src.dialogue_manager import DialogueManager
from src.ui import print_user_input, print_assistant_response, print_dialogue_context, print_sentiment_analysis
from src.utils.security import SecurityManager
from src.utils.error_handler import error_handler, logger, AIAssistantException

# 加载 .env 文件
load_dotenv()

def print_help() -> None:
    """打印帮助信息"""
    print("\n可用命令：")
    print("'quit' - 退出程序")
    print("'clear' - 清除对话历史")
    print("'sentiment' - 对下一条输入进行情感分析")
    print("'execute' - 执行 Python 代码")
    print("'summarize' - 生成文本摘要")
    print("'rl' - 使用强化学习")
    print("'auto_fe' - 使用自动特征工程")
    print("'interpret' - 使用模型解释性")
    print("'active_learn' - 使用主动学习")
    print("'change_model' - 更改使用的模型")
    print("'help' - 显示此帮助信息")

class AINirvana:
    def __init__(self, config: Config):
        self.config = config
        self.api_key = os.getenv('OPENAI_API_KEY') or self.config.get('api_key')
        self.model_name = self.config.get('model', 'gpt-3.5-turbo')
        self.use_gpu = self.config.get('use_gpu', False)
        self.system_prompt = self.config.get('system_prompt', '')

        self.assistant = AIAssistant(
            model_name=self.model_name,
            use_gpu=self.use_gpu,
            system_prompt=self.system_prompt,
            api_key=self.api_key
        )
        self.dialogue_manager = DialogueManager(max_history=self.config.get('max_context_length', 5))
        self.security_manager = SecurityManager()

    @error_handler
    def process(self, input_text: str) -> str:
        """处理用户输入并生成响应"""
        context = self.dialogue_manager.get_dialogue_context()
        response = self.assistant.generate_response(input_text, context)
        self.dialogue_manager.add_to_history(input_text, response)
        return response

def handle_command(command: str, ai_nirvana: AINirvana) -> Dict[str, Any]:
    """处理特殊命令"""
    if command == 'clear':
        ai_nirvana.assistant.clear_context()
        ai_nirvana.dialogue_manager.clear_history()
        return {"message": "对话历史已清除。", "continue": True}
    elif command == 'help':
        print_help()
        return {"continue": True}
    elif command == 'quit':
        return {"message": "谢谢使用，再见！", "continue": False}
    return {"continue": True}

@error_handler
def main(config: Config):
    ai_nirvana = AINirvana(config)
    logger.info("AI Nirvana 智能助手启动")
    print("欢迎使用 AI Nirvana 智能助手！")
    print("输入 'help' 查看可用命令。")

    while True:
        try:
            user_input = input("\n请输入您的问题或文本：\n").strip().lower()
            
            command_result = handle_command(user_input, ai_nirvana)
            if not command_result.get("continue", True):
                print(command_result.get("message", ""))
                break
            
            if command_result.get("message"):
                print(command_result["message"])
                continue

            response = ai_nirvana.process(user_input)
            print_user_input(user_input)
            print("\n回答：")
            print_assistant_response(response)
            print_dialogue_context(ai_nirvana.dialogue_manager.get_dialogue_context())

        except AIAssistantException as e:
            logger.error(f"AI Assistant error: {str(e)}")
            print(f"发生错误: {str(e)}")
        except Exception as e:
            logger.exception(f"Unexpected error: {str(e)}")
            print(f"发生意外错误: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Nirvana Assistant")
    parser.add_argument("--config", default="config.json", help="Path to the configuration file")
    args = parser.parse_args()

    config = Config(args.config)
    if not config.validate_config():
        logger.error("Configuration validation failed. Please check your config file.")
        sys.exit(1)

    main(config)