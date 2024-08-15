from src.core.enhanced_ai_assistant import EnhancedAIAssistant

def main():
    assistant = EnhancedAIAssistant()

    # 示例 1: 检测语言并翻译
    text = "Bonjour, comment ça va?"
    detected_lang = assistant.detect_language(text)
    print(f"检测到的语言: {detected_lang}")
    
    translated_text = assistant.translate(text, detected_lang, 'en')
    print(f"翻译结果: {translated_text}")

    # 示例 2: 生成响应
    prompt = "What is the capital of France?"
    response = assistant.generate_response(prompt)
    print(f"生成的响应: {response}")

    # 示例 3: 情感分析
    sentiment_text = "I love this product! It's amazing!"
    sentiment = assistant.analyze_sentiment(sentiment_text)
    print(f"情感分析结果: {sentiment}")

    # 示例 4: 提取关键词
    keyword_text = "Artificial intelligence is transforming various industries including healthcare and finance."
    keywords = assistant.extract_keywords(keyword_text)
    print(f"提取的关键词: {keywords}")

if __name__ == "__main__":
    main()