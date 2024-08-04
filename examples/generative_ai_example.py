from src.core.generative_ai import GenerativeAI
from PIL import Image

def main():
    # 初始化 GenerativeAI
    ai = GenerativeAI()

    # 文本生成示例
    prompt = "The quick brown fox"
    generated_text = ai.generate_text(prompt)
    print("Generated Text:", generated_text)

    # 文本翻译示例
    text = "Hello, world!"
    translated_text = ai.translate_text(text)
    print("Translated Text:", translated_text)

    # 图像分类示例
    image_path = "path/to/your/image.jpg"
    classification = ai.classify_image(image_path)
    print("Image Classification:", classification)

    # 图像描述生成示例
    image = Image.open(image_path)
    caption = ai.generate_image_caption(image)
    print("Image Caption:", caption)

    # 问答示例
    context = "The capital of France is Paris. It is known for its beautiful architecture and cuisine."
    question = "What is the capital of France?"
    answer = ai.answer_question(context, question)
    print("Answer:", answer)

    # 情感分析示例
    sentiment_text = "I love this product! It's amazing!"
    sentiment = ai.analyze_sentiment(sentiment_text)
    print("Sentiment:", sentiment)

    # 文本摘要示例
    long_text = "Long text to be summarized..." * 10
    summary = ai.summarize_text(long_text)
    print("Summary:", summary)

    # 微调示例
    train_data = ["Example text 1", "Example text 2", "Example text 3"]
    ai.fine_tune(train_data, epochs=1)

    # 保存模型示例
    ai.save_model("path/to/save/model")

    # 加载模型示例
    ai.load_model("path/to/saved/model")

if __name__ == "__main__":
    main()