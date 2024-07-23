import os  
import sys  
import argparse  
from typing import Dict, Any, List  
from dotenv import load_dotenv  
from io import StringIO  

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  

from src.config import Config  
from src.core.ai_assistant import AIAssistant  
from src.dialogue_manager import DialogueManager  
from src.ui import print_user_input, print_assistant_response, print_dialogue_context, print_sentiment_analysis  
from src.utils.security import SecurityManager  
from src.utils.error_handler import error_handler, logger, AIAssistantException  

# 加载 .env 文件  
load_dotenv()  

print("Script started")  

def process_input(input_text: str, config: Config) -> str:  
    ai_nirvana = AINirvana(config)  
    return ai_nirvana.process(input_text)  

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

def execute_code(code):  
    old_stdout = sys.stdout  
    redirected_output = sys.stdout = StringIO()  
    try:  
        exec(code)  
        sys.stdout = old_stdout  
        return redirected_output.getvalue()  
    except Exception as e:  
        sys.stdout = old_stdout  
        return f"执行错误: {str(e)}"  

def handle_command(command: str, ai_nirvana: AINirvana) -> Dict[str, Any]:  
    """处理特殊命令"""  
    if command == 'clear':  
        message = ai_nirvana.dialogue_manager.clear_history()  
        ai_nirvana.assistant.clear_context()  
        return {"message": message + " 输入新的问题开始新的对话。", "continue": True}  
    elif command == 'help':  
        print_help()  
        return {"message": "如需更多帮助，请具体描述您的问题。", "continue": True}  
    elif command == 'quit':  
        return {"message": "谢谢使用 AI Nirvana 智能助手，再见！", "continue": False}  
    elif command == 'sentiment':  
        text = input("请输入要分析情感的文本：")  
        sentiment = ai_nirvana.analyze_sentiment(text)  
        print_sentiment_analysis(sentiment)  
        print("情感分析结果解释：")  
        print(f"积极情绪: {sentiment['positive']:.2f}")  
        print(f"中性情绪: {sentiment['neutral']:.2f}")  
        print(f"消极情绪: {sentiment['negative']:.2f}")  
        return {"continue": True}  
    elif command == 'summarize':  
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
        return {"message": "摘要生成完成。需要进一步解释或重新生成吗？", "continue": True}  
    elif command == 'change_model':  
        while True:  
            available_models = ai_nirvana.get_available_models()  
            print(f"可用的模型有：{', '.join(available_models)}")  
            model_name = input("请输入新的模型名称（或输入 'cancel' 取消）：").strip().lower()  
            if model_name == 'cancel':  
                return {"message": "已取消更改模型。", "continue": True}  
            if model_name not in [m.lower() for m in available_models]:  
                print(f"错误：'{model_name}' 不是有效的模型名称。请检查拼写并重试。")  
                continue  
            try:  
                # 使用原始大小写的模型名称  
                original_case_model_name = next(m for m in available_models if m.lower() == model_name)  
                ai_nirvana.change_model(original_case_model_name)  
                return {"message": f"模型已更改为 {original_case_model_name}。试试问一个问题吧！", "continue": True}  
            except Exception as e:  
                print(f"更改模型失败: {str(e)}。请重试或选择其他模型。")  
    elif command == 'execute':  
        print("请输入要执行的 Python 代码（输入空行结束）：")  
        lines = []  
        while True:  
            line = input()  
            if line.strip() == "":  
                break  
            lines.append(line)  
        code = "\n".join(lines)  
        result = execute_code(code)  
        return {"message": f"执行结果:\n{result}\n需要解释结果吗？", "continue": True}  
    else:  
        # 处理一般文本输入  
        response = ai_nirvana.process(command)  
        print_user_input(command)  
        print("\n回答：")  
        print_assistant_response(response)  
        print_dialogue_context(ai_nirvana.dialogue_manager.get_dialogue_context())  
        return {"continue": True}  

@error_handler  
def main(config: Config):  
    print("Main function started")  
    ai_nirvana = AINirvana(config)  
    logger.info("AI Nirvana 智能助手启动")  
    print("欢迎使用 AI Nirvana 智能助手！")  
    print("输入 'help' 查看可用命令。")  

    while True:  
        try:  
            print("\n请输入您的问题或文本（输入空行发送，输入特殊命令如 'help' 直接执行）：")  
            lines = []  
            while True:  
                try:  
                    line = input()  
                    if line.strip() == "":  
                        break  
                    lines.append(line)  
                except EOFError:  
                    print("检测到 EOF，退出程序")  
                    return  
            user_input = "\n".join(lines)  
            
            if not user_input.strip():  
                print("您似乎没有输入任何内容。请输入一些文字或命令。")  
                continue  
            
            if user_input.lower() in ['help', 'clear', 'quit', 'sentiment', 'summarize', 'change_model', 'execute']:  
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

        print(f"Loaded configuration: {config.config}")  

        main(config)  
    except Exception as e:  
        print(f"An unexpected error occurred: {str(e)}")  
        import traceback  
        print(traceback.format_exc())  

# 添加服务器启动代码  
from flask import Flask, request, jsonify  

app = Flask(__name__)  

@app.route('/process', methods=['POST'])  
def process():  
    input_text = request.json.get('input')  
    config = Config("config.json")  
    response = process_input(input_text, config)  
    return jsonify({"response": response})  

if __name__ == "__main__":  
    print("Starting Flask server...")  
    app.run(host='0.0.0.0', port=8000)  

__all__ = ['process_input', 'AINirvana', 'Config']