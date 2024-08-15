import logging

logger = logging.getLogger(__name__)

def test_model_comparison(ai_nirvana):
    test_text = "人工智能正在改变我们的生活方式，从智能家居到自动驾驶汽车，AI无处不在。"
    models = ["gpt-3.5-turbo", "gpt-4"]

    for model in models:
        try:
            ai_nirvana.change_model(model)
            print(f"\n使用模型 {model} 的结果：")
            
            # 摘要测试
            try:
                summary = ai_nirvana.summarize(test_text)
                print(f"摘要： {summary}")
            except Exception as e:
                logger.error(f"摘要生成错误: {str(e)}")
                print(f"摘要生成错误: {str(e)}")

            # 情感分析测试
            try:
                sentiment = ai_nirvana.analyze_sentiment(test_text)
                print(f"情感分析： {sentiment}")
            except Exception as e:
                logger.error(f"情感分析错误: {str(e)}")
                print(f"情感分析错误: {str(e)}")

            # 关键词提取测试
            try:
                keywords = ai_nirvana.extract_keywords(test_text)
                print(f"关键词提取： {keywords}")
            except Exception as e:
                logger.error(f"关键词提取错误: {str(e)}")
                print(f"关键词提取错误: {str(e)}")

        except Exception as e:
            logger.error(f"使用模型 {model} 时发生错误: {str(e)}")
            print(f"使用模型 {model} 时发生错误: {str(e)}")

    return {"continue": True}