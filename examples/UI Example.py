from src.ui import (
    print_user_input,
    print_assistant_response,
    print_dialogue_context,
    print_sentiment_analysis,
)


def main():
    # 模拟用户输入
    user_input = "Hello, AI! How are you today?"
    print_user_input(user_input)

    # 模拟 AI 助手响应
    assistant_response = "Hello! As an AI, I don't have feelings, but I'm functioning well and ready to assist you. How can I help you today?"
    print_assistant_response(assistant_response)

    # 模拟对话上下文
    dialogue_context = """
User: Hello, AI! How are you today?
Assistant: Hello! As an AI, I don't have feelings, but I'm functioning well and ready to assist you. How can I help you today?
User: Can you tell me about the weather?
Assistant: I'm sorry, but I don't have access to real-time weather information. You might want to check a weather website or app for the most up-to-date weather data. Is there anything else I can help you with?
"""
    print_dialogue_context(dialogue_context)

    # 模拟情感分析结果
    sentiment = {"positive": 0.7, "neutral": 0.2, "negative": 0.1}
    print_sentiment_analysis(sentiment)


if __name__ == "__main__":
    main()
