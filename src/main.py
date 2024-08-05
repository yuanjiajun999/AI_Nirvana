import os
import sys
import argparse
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from typing import Dict, Any, List
from io import StringIO

# 调整路径以便于模块导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.core.ai_assistant import AIAssistant
from src.core.generative_ai import GenerativeAI
from src.core.multimodal import MultimodalInterface
from src.utils.error_handler import error_handler, logger, AIAssistantException
from src.utils.security import SecurityManager
from src.dialogue_manager import DialogueManager
from src.ui import print_user_input, print_assistant_response, print_dialogue_context, print_sentiment_analysis

# 加载 .env 文件
load_dotenv()

class AINirvana:
    def __init__(self, config: Config):
        self.config = config
        self.model_name = self.config.get('model', 'gpt-3.5-turbo')
        self.max_context_length = self.config.get('max_context_length', 5)
        self.assistant = AIAssistant(model_name=self.model_name, max_context_length=self.max_context_length)
        self.generative_ai = GenerativeAI()
        self.multimodal_interface = MultimodalInterface()
        self.dialogue_manager = DialogueManager(max_history=self.max_context_length)
        self.security_manager = SecurityManager()
        self.variable_state = {}  # 添加变量状态存储
        print(f"AINirvana instance created with model: {self.model_name}")

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

    def get_available_models(self) -> List[str]:
        return self.assistant.get_available_models()

    def execute_code(self, code: str) -> str:
        """在受限环境中执行Python代码并返回输出"""
        old_stdout = sys.stdout
        redirected_output = StringIO()
        sys.stdout = redirected_output

        # 创建一个本地命名空间来存储变量
        local_vars = self.variable_state.copy()

        # 创建一个受限的环境
        safe_dict = {
            'abs': abs, 'all': all, 'any': any, 'ascii': ascii, 
            'bin': bin, 'bool': bool, 'chr': chr, 'dict': dict, 
            'dir': dir, 'divmod': divmod, 'enumerate': enumerate, 
            'filter': filter, 'float': float, 'format': format, 
            'frozenset': frozenset, 'hash': hash, 'hex': hex, 
            'int': int, 'isinstance': isinstance, 'issubclass': issubclass, 
            'len': len, 'list': list, 'map': map, 'max': max, 
            'min': min, 'oct': oct, 'ord': ord, 'pow': pow, 
            'print': print, 'range': range, 'repr': repr, 
            'reversed': reversed, 'round': round, 'set': set, 
            'slice': slice, 'sorted': sorted, 'str': str, 
            'sum': sum, 'tuple': tuple, 'type': type, 'zip': zip,
            'ZeroDivisionError': ZeroDivisionError,  # 添加常见异常
            'ValueError': ValueError,
            'TypeError': TypeError
        }

        try:
            # 在受限环境中执行代码
            exec(code, {"__builtins__": safe_dict}, local_vars)
            output = redirected_output.getvalue()
            
            # 显示更新后的变量值
            for var, value in local_vars.items():
                if not var.startswith('__'):
                    output += f"\n{var} = {value} (Type: {type(value).__name__})"
            
            # 更新变量状态
            self.variable_state.update(local_vars)
            
            return output
        except Exception as e:
            return f"执行错误: {type(e).__name__} - {str(e)}"
        finally:
            sys.stdout = old_stdout

def handle_command(command: str, ai_nirvana: AINirvana) -> Dict[str, Any]:
    """处理用户输入的命令"""
    if command == "clear":
        ai_nirvana.dialogue_manager.clear_history()
        ai_nirvana.assistant.clear_context()
        ai_nirvana.variable_state.clear()  # 清除变量状态
        return {"message": "对话历史和变量状态已清除。", "continue": True}
    elif command == "help":
        print_help()
        return {"message": "请按照以上帮助信息操作。", "continue": True}
    elif command == "quit":
        return {"message": "谢谢使用 AI Nirvana 智能助手，再见！", "continue": False}
    elif command == "sentiment":
        text = input("请输入要分析情感的文本：")
        sentiment = ai_nirvana.analyze_sentiment(text)
        print_sentiment_analysis(sentiment)
        return {"continue": True}
    elif command == "summarize":
        print("请输入要生成摘要的文本（输入空行结束）：")
        lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        text = "\n".join(lines)
        summary = ai_nirvana.summarize(text)
        print(f"摘要：{summary}")
        return {"continue": True}
    elif command == "change_model":
        available_models = ai_nirvana.get_available_models()
        print(f"可用的模型有：{', '.join(available_models)}")
        model_name = input("请输入新的模型名称（或输入 'cancel' 取消）：").strip()
        if model_name == "cancel":
            return {"message": "已取消更改模型。", "continue": True}
        if model_name in available_models:
            ai_nirvana.change_model(model_name)
            return {"message": f"模型已更改为 {model_name}。", "continue": True}
        else:
            print(f"无效的模型名称：{model_name}")
            return {"continue": True}
    elif command == "execute":
        print("请输入要执行的 Python 代码（输入空行结束）：")
        lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        code = "\n".join(lines)
        result = ai_nirvana.execute_code(code)
        print("执行结果:")
        print(result)
        return {"message": "需要进一步解释结果吗？", "continue": True}
    elif command == "vars":
        variables = ai_nirvana.variable_state
        print("当前定义的变量:")
        for var, value in variables.items():
            print(f"{var} = {value} (Type: {type(value).__name__})")
        return {"continue": True}
    else:
        response = ai_nirvana.process(command)
        print_user_input(command)
        print("\n回答：")
        print_assistant_response(response)
        print_dialogue_context(ai_nirvana.dialogue_manager.get_dialogue_context())
        return {"continue": True}

def print_help() -> None:
    """打印帮助信息"""
    print("\n可用命令：")
    print("'quit' - 退出程序")
    print("'clear' - 清除对话历史和变量状态")
    print("'sentiment' - 对下一条输入进行情感分析")
    print("'execute' - 执行 Python 代码")
    print("'summarize' - 生成文本摘要")
    print("'change_model' - 更改使用的模型")
    print("'vars' - 显示当前定义的所有变量")
    print("'help' - 显示此帮助信息")
    print("\n注意：")
    print("- 执行代码时，某些操作（如文件操作和模块导入）出于安全考虑是受限的。")
    print("- 支持基本的Python操作，包括变量赋值、条件语句、循环等。")
    print("- 如果遇到'未定义'的错误，可能是因为该操作被安全限制所阻止。")

@error_handler
def main(config: Config):
    print("欢迎使用 AI Nirvana 智能助手！")
    print("输入 'help' 查看可用命令。")
    ai_nirvana = AINirvana(config)

    while True:
        try:
            print("\n请输入您的问题或命令（输入空行发送）：")
            lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                lines.append(line)
            user_input = "\n".join(lines)

            if not user_input.strip():
                print("您似乎没有输入任何内容。请输入一些文字或命令。")
                continue

            if user_input.lower() in ["help", "clear", "quit", "sentiment", "summarize", "change_model", "execute", "vars"]:
                command_result = handle_command(user_input.lower(), ai_nirvana)
            else:
                command_result = handle_command(user_input, ai_nirvana)

            if not command_result.get("continue", True):
                print(command_result.get("message", ""))
                break

            if command_result.get("message"):
                print(command_result["message"])

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

    try:
        config = Config(args.config)
        if not config.validate_config():
            logger.error("Configuration validation failed. Please check your config file.")
            sys.exit(1)

        main(config)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

# 添加Flask服务器代码
app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process():
    input_text = request.json.get("input")
    config = Config("config.json")
    ai_nirvana = AINirvana(config)
    response = ai_nirvana.process(input_text)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)