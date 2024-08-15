EnhancedAIAssistant 使用指南
EnhancedAIAssistant 是一个功能强大的 AI 助手类，集成了多种自然语言处理能力。本指南将帮助您了解如何使用 EnhancedAIAssistant 的主要功能。
1. 初始化
首先，导入 EnhancedAIAssistant 类并创建一个实例：
from src.core.enhanced_ai_assistant import EnhancedAIAssistant

assistant = EnhancedAIAssistant()
2. 语言检测和翻译
检测语言
text = "Bonjour, comment ça va?"
detected_lang = assistant.detect_language(text)
print(f"检测到的语言: {detected_lang}")
翻译文本
translated_text = assistant.translate(text, detected_lang, 'en')
print(f"翻译结果: {translated_text}")
3. 生成响应
使用 AI 生成对给定提示的响应：
prompt = "What is the capital of France?"
response = assistant.generate_response(prompt)
print(f"生成的响应: {response}")
4. 情感分析
分析文本的情感：
sentiment_text = "I love this product! It's amazing!"
sentiment = assistant.analyze_sentiment(sentiment_text)
print(f"情感分析结果: {sentiment}")
5. 提取关键词
从文本中提取关键词：
keyword_text = "Artificial intelligence is transforming various industries including healthcare and finance."
keywords = assistant.extract_keywords(keyword_text)
print(f"提取的关键词: {keywords}")
注意事项

确保在使用前正确设置了所有必要的依赖和环境变量。
对于较长的文本或大量请求，考虑性能影响和 API 调用限制。
某些功能可能需要互联网连接（如翻译和某些 AI 模型）。

高级功能
EnhancedAIAssistant 还提供了其他高级功能，如代码执行、任务规划等。请参考类的文档字符串或源代码以了解更多详细信息。

这个示例文件和文档提供了 EnhancedAIAssistant 类的基本用法。您可以根据需要扩展示例或文档，以涵盖更多特定的使用场景或功能。