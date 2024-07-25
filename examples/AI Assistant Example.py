from src.core.ai_assistant import AIAssistant


def main():
    assistant = AIAssistant()

    # 生成响应
    prompt = "Tell me about artificial intelligence."
    response = assistant.generate_response(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")

    # 文本摘要
    long_text = "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. Leading AI textbooks define the field as the study of 'intelligent agents': any system that perceives its environment and takes actions that maximize its chance of achieving its goals."
    summary = assistant.summarize(long_text)
    print("Original text:", long_text)
    print(f"Summary: {summary}\n")

    # 情感分析
    text = "I love using this AI assistant! It's so helpful and efficient."
    sentiment = assistant.analyze_sentiment(text)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}\n")

    # 更改模型
    assistant.change_model("gpt-4")
    print("Model changed to GPT-4")

    # 加密敏感数据
    sensitive_data = "This is confidential information."
    encrypted_data = assistant.encrypt_sensitive_data(sensitive_data)
    print(f"Encrypted data: {encrypted_data}")
    decrypted_data = assistant.decrypt_sensitive_data(encrypted_data)
    print(f"Decrypted data: {decrypted_data}\n")

    # 执行代码
    code = "print('Hello, World!')"
    result, error = assistant.execute_code(code, "python")
    print(f"Code execution result: {result}")
    print(f"Code execution error: {error}")


if __name__ == "__main__":
    main()
