def test_complex_text_analysis(ai_nirvana):
    # 多段落摘要
    long_text = """
    人工智能（AI）是计算机科学的一个分支，致力于创造智能机器。它已经成为现代技术中最令人兴奋的领域之一。AI的应用范围从简单的智能助手到复杂的自动驾驶系统。

    机器学习是AI的一个核心组成部分。它允许系统从数据中学习，而无需明确编程。深度学习，作为机器学习的一个子集，使用神经网络来模仿人脑的工作方式。

    AI在医疗保健、金融、教育等多个领域都有重要应用。例如，在医疗领域，AI可以帮助诊断疾病，分析医学图像，甚至协助手术。在金融领域，AI被用于预测市场趋势，检测欺诈行为。

    然而，AI的发展也带来了一些担忧。隐私问题、就业替代、以及AI系统的偏见都是需要解决的重要问题。因此，负责任地发展AI技术变得越来越重要。

    展望未来，AI有望继续改变我们的生活和工作方式。但我们必须确保这种变革是以造福人类的方式进行的。
    """
    print("多段落摘要测试：")
    print(ai_nirvana.summarize(long_text))
    print("\n")
    
    summary = ai_nirvana.summarize(long_text)
    print(f"摘要：{summary}")
    
    # 多样本情感分析
    samples = [
        "今天是个好天气，我心情很愉快。",
        "这部电影真是太无聊了，浪费了我两个小时。",
        "虽然考试没考好，但我会继续努力的。",
        "新工作很有挑战性，我既兴奋又紧张。",
        "听到这个消息，我既高兴又有点担心。"
    ]
    print("多样本情感分析测试：")
    for sample in samples:
        try:
            sentiment = ai_nirvana.analyze_sentiment(sample)
            print(f"文本：{sample}")
            print(f"情感分析结果：{sentiment}")
            print(f"API 响应：{ai_nirvana.assistant.language_model.last_response}")  # 假设您在 LanguageModel 类中添加了 last_response 属性
        except Exception as e:
            print(f"分析文本 '{sample}' 时发生错误: {str(e)}")
        print()

    print("Complex text analysis test function")
    return True    