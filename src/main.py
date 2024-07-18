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
    print("'change_model' - 更改使用的模型")
    print("'help' - 显示此帮助信息")

class AINirvana:
    def __init__(self, config: Config):
        self.config = config
        self.model_name = self.config.get('model', 'gpt-3.5-turbo')
        self.max_context_length = self.config.get('max_context_length', 5)

        self.assistant = AIAssistant(model_name=self.model_name, max_context_length=self.max_context_length)
        self.dialogue_manager = DialogueManager(max_history=self.max_context_length)
        self.security_manager = SecurityManager()

    @error_handler
    def process(self, input_text: str) -> str:
        """处理用户输入并生成响应"""
        if not self.security_manager.is_safe_code(input_text):
            raise AIAssistantException("Unsafe input detected")
        response = self.assistant.generate_response(input_text)
        self.dialogue_manager.add_to_history(input_text, response)
        return response

    def summarize(self, text: str) -> str:
        return self.assistant.summarize(text)

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        return self.assistant.analyze_sentiment(text)

    def change_model(self, model_name: str) -> None:
        self.assistant.change_model(model_name)

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
    elif command == 'sentiment':
        text = input("请输入要分析情感的文本：")
        sentiment = ai_nirvana.analyze_sentiment(text)
        print_sentiment_analysis(sentiment)
        return {"continue": True}
    elif command == 'summarize':
        text = input("请输入要生成摘要的文本：")
        summary = ai_nirvana.summarize(text)
        print(f"摘要：{summary}")
        return {"continue": True}
    elif command == 'change_model':
        model_name = input("请输入新的模型名称：")
        ai_nirvana.change_model(model_name)
        return {"message": f"模型已更改为 {model_name}", "continue": True}
    return {"continue": True}

@error_handler
def main(config: Config):
    ai_nirvana = AINirvana(config)
    logger.info("AI Nirvana 智能助手启动")
    print("欢迎使用 AI Nirvana 智能助手！")
    print("输入 'help' 查看可用命令。")

    while True:
        try:
            user_input = input("\n请输入您的问题或文本：\n").strip()
            
            command_result = handle_command(user_input.lower(), ai_nirvana)
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