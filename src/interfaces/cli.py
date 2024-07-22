from src.core.ai_assistant import AIAssistant
from src.dialogue_manager import DialogueManager
from src.ui import (print_assistant_response, print_dialogue_context,
                    print_sentiment_analysis, print_user_input)


def print_help():
    print("\n可用命令：")
    print("'quit' - 退出程序")
    print("'clear' - 清除对话历史")
    print("'sentiment' - 对下一条输入进行情感分析")
    print("'help' - 显示此帮助信息")
    print("长文本（超过设定长度）会自动进行摘要")

def run_cli(assistant: AIAssistant, dialogue_manager: DialogueManager, max_input_length: int = 100):
    print("欢迎使用 AI Nirvana 智能助手！")
    print("输入 'help' 查看可用命令。")

    analyze_sentiment = False

    while True:
        user_input = input("\n请输入您的问题或文本：\n")

        if user_input.lower() == 'quit':
            print("谢谢使用，再见！")
            break
        elif user_input.lower() == 'clear':
            assistant.clear_context()
            dialogue_manager.clear_history()
            print("对话历史已清除。")
            continue
        elif user_input.lower() == 'help':
            print_help()
            continue
        elif user_input.lower() == 'sentiment':
            analyze_sentiment = True
            print("下一条输入将进行情感分析。")
            continue

        try:
            if analyze_sentiment:
                sentiment = assistant.analyze_sentiment(user_input)
                print_user_input(user_input)
                print("\n情感分析结果：")
                print_sentiment_analysis(sentiment)
                analyze_sentiment = False
            elif len(user_input) > max_input_length:
                response = assistant.summarize(user_input)
                print_user_input(user_input)
                print("\n摘要：")
                print_assistant_response(response)
            else:
                response = assistant.generate_response(user_input)
                print_user_input(user_input)
                print("\n回答：")
                print_assistant_response(response)

            dialogue_manager.add_to_history(user_input, response)
            print_dialogue_context(dialogue_manager.get_dialogue_context())
        except Exception as e:
            print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    # This is just for testing the CLI independently
    from src.core.ai_assistant import AIAssistant
    from src.dialogue_manager import DialogueManager
    
    assistant = AIAssistant()
    dialogue_manager = DialogueManager()
    run_cli(assistant, dialogue_manager)