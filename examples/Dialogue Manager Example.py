from src.dialogue_manager import DialogueManager


def main():
    dialogue_manager = DialogueManager(max_history=5)

    # 添加对话历史
    dialogue_manager.add_to_history(
        "User: Hello, AI!", "AI: Hello! How can I assist you today?"
    )
    dialogue_manager.add_to_history(
        "User: What's the weather like?",
        "AI: I'm sorry, I don't have real-time weather information. You might want to check a weather app or website for the most up-to-date information.",
    )
    dialogue_manager.add_to_history(
        "User: Can you tell me a joke?",
        "AI: Sure! Here's one: Why don't scientists trust atoms? Because they make up everything!",
    )

    # 获取对话上下文
    context = dialogue_manager.get_dialogue_context()
    print("Current dialogue context:")
    print(context)

    # 添加更多对话，超过最大历史限制
    dialogue_manager.add_to_history(
        "User: That's funny!",
        "AI: I'm glad you enjoyed it! Do you have any other questions?",
    )
    dialogue_manager.add_to_history(
        "User: No, that's all for now.",
        "AI: Alright! If you need any more help, feel free to ask. Have a great day!",
    )
    dialogue_manager.add_to_history(
        "User: Actually, one more thing...",
        "AI: Of course! What would you like to know?",
    )

    # 再次获取对话上下文，查看是否保持了最大历史限制
    context = dialogue_manager.get_dialogue_context()
    print("\nUpdated dialogue context (should only show last 5 interactions):")
    print(context)

    # 清除对话历史
    clear_message = dialogue_manager.clear_history()
    print(f"\n{clear_message}")

    # 确认历史已被清除
    context = dialogue_manager.get_dialogue_context()
    print("Dialogue context after clearing:")
    print(context)


if __name__ == "__main__":
    main()
