from src.core.langchain import get_response, LangChainAgent


def main():
    # 使用 get_response 函数
    question = "What is the capital of France?"
    response = get_response(question)
    print(f"Question: {question}")
    print(f"Response: {response}\n")

    # 使用 LangChainAgent
    agent = LangChainAgent()

    # QA 任务
    qa_query = "Explain the concept of machine learning in simple terms."
    qa_response = agent.run_qa_task(qa_query)
    print(f"QA Task Query: {qa_query}")
    print(f"QA Task Response: {qa_response}\n")

    # 摘要任务
    text_to_summarize = "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. Leading AI textbooks define the field as the study of 'intelligent agents': any system that perceives its environment and takes actions that maximize its chance of achieving its goals."
    summary = agent.run_summarization_task(text_to_summarize)
    print("Summarization Task:")
    print(f"Original Text: {text_to_summarize}")
    print(f"Summary: {summary}\n")

    # 生成任务
    generation_prompt = "Write a short poem about artificial intelligence."
    generated_text = agent.run_generation_task(generation_prompt)
    print(f"Generation Task Prompt: {generation_prompt}")
    print(f"Generated Text: {generated_text}")


if __name__ == "__main__":
    main()
