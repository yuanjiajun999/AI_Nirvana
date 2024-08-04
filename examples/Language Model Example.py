from src.core.language_model import LanguageModel

def main():
    # 初始化语言模型
    lm = LanguageModel()

    # 生成响应
    prompt = "What is artificial intelligence?"
    response = lm.generate_response(prompt)
    print(f"Response: {response}\n")

    # 获取可用模型
    models = lm.get_available_models()
    print(f"Available models: {models}\n")

    # 更改默认模型
    lm.change_default_model("gpt-4")
    print(f"Default model changed to: {lm.default_model}\n")

    # 获取模型信息
    model_info = lm.get_model_info()
    print(f"Model info: {model_info}\n")

    # 分析情感
    text = "I love using this language model! It's amazing!"
    sentiment = lm.analyze_sentiment(text)
    print(f"Sentiment analysis: {sentiment}\n")

    # 生成摘要
    long_text = "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals."
    summary = lm.summarize(long_text)
    print(f"Summary: {summary}\n")

    # 翻译文本
    text_to_translate = "Hello, how are you?"
    translated_text = lm.translate(text_to_translate, "French")
    print(f"Translated text: {translated_text}")

if __name__ == "__main__":
    main()