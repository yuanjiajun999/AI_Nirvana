from src.core.api_model import APIModel, ConcreteAPIModel


def main():
    # 创建一个具体的 API 模型实例
    api_model = ConcreteAPIModel(
        api_key="your_api_key", api_url="https://api.example.com/generate"
    )

    # 生成响应
    prompt = "Tell me a joke about programming."
    response = api_model.generate_response(prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated response: {response}\n")

    # 使用缓存功能
    cached_response = api_model.generate(prompt)
    print(f"Cached response: {cached_response}\n")

    # 生成文本摘要
    long_text = "This is a long piece of text that needs to be summarized. It contains many sentences and covers various topics. The goal is to create a concise summary of the main points."
    summary = api_model.summarize(long_text)
    print(f"Original text: {long_text}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
