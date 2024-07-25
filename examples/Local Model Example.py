from src.core.local_model import LocalModel


def main():
    local_model = LocalModel()

    # 生成响应
    prompts = [
        "你是谁",
        "你能做什么",
        "生成式 AI",
        "AI 发展",
        "这是一个不在预定义响应中的问题",
    ]

    for prompt in prompts:
        response = local_model.generate_response(prompt)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}\n")

    # 生成摘要
    text = "这是一个很长的文本，需要进行摘要。它包含了许多句子和各种主题。我们的目标是创建一个简洁的摘要，概括主要内容。"
    summary = local_model.summarize(text)
    print(f"Original text: {text}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
