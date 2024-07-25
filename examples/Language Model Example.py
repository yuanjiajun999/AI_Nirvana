from src.core.language_model import LanguageModel


def main():
    model = LanguageModel()

    # 生成响应
    prompt = "Explain the concept of artificial intelligence."
    response = model.generate_response(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")

    # 获取可用模型
    available_models = model.get_available_models()
    print(f"Available models: {available_models}\n")

    # 更改默认模型
    new_model = "gpt-4"  # 假设这是一个可用的模型
    model.change_default_model(new_model)
    print(f"Changed default model to: {new_model}\n")

    # 获取模型信息
    model_info = model.get_model_info()
    print(f"Model info: {model_info}\n")

    # 情感分析
    text = "I love using this AI system!"
    sentiment = model.analyze_sentiment(text)
    print(f"Text: {text}")
    print(f"Sentiment analysis: {sentiment}")


if __name__ == "__main__":
    main()
