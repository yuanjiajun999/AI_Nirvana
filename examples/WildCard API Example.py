from src.core.wildcard_api import WildCardAPI


def main():
    api = WildCardAPI("your_api_key_here")

    # 聊天补全示例
    chat_messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What's the weather like today?"},
    ]
    chat_response = api.chat_completion("gpt-3.5-turbo", chat_messages)
    print("Chat Completion Response:", chat_response)

    # 嵌入示例
    text = "The quick brown fox jumps over the lazy dog."
    embeddings = api.embeddings("text-embedding-ada-002", text)
    print("Embeddings:", embeddings)

    # 图像生成示例
    prompt = "A beautiful sunset over a mountain lake"
    images = api.image_generation("dall-e-3", prompt, n=1, size="1024x1024")
    print("Generated Image URLs:", images)


if __name__ == "__main__":
    main()
