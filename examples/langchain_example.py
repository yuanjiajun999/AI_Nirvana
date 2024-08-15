from src.core.langchain import LangChainAgent

def main():
    # 创建 LangChainAgent 实例
    agent = LangChainAgent()

    # 问答任务
    question = "What is the capital of France?"
    answer = agent.run_qa_task(question)
    print(f"Q: {question}\nA: {answer}\n")

    # 摘要任务
    text_to_summarize = "Artificial intelligence (AI) is intelligence demonstrated by machines, " \
                        "as opposed to natural intelligence displayed by animals including humans. " \
                        "AI research has been defined as the field of study of intelligent agents, " \
                        "which refers to any system that perceives its environment and takes actions " \
                        "that maximize its chance of achieving its goals."
    summary = agent.run_summarization_task(text_to_summarize)
    print(f"Original text: {text_to_summarize}\nSummary: {summary}\n")

    # 文本生成任务
    prompt = "Write a short story about a robot learning to feel emotions."
    generated_text = agent.run_generation_task(prompt)
    print(f"Prompt: {prompt}\nGenerated text: {generated_text}\n")

    # 情感分析任务
    text_to_analyze = "I absolutely love this new restaurant! The food is amazing and the service is top-notch."
    sentiment = agent.analyze_sentiment(text_to_analyze)
    print(f"Text: {text_to_analyze}\nSentiment: {sentiment}\n")

    # 关键词提取任务
    text_for_keywords = "Machine learning is a subset of artificial intelligence that focuses on the development " \
                        "of algorithms and statistical models that enable computer systems to improve their performance " \
                        "on a specific task through experience."
    keywords = agent.extract_keywords(text_for_keywords)
    print(f"Text: {text_for_keywords}\nKeywords: {keywords}")

if __name__ == "__main__":
    main()