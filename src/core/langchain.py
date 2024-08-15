import os
from functools import lru_cache
from typing import Any, Dict, Union, Optional

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_openai import ChatOpenAI
from openai import OpenAIError

load_dotenv()

class APIConfig:
    API_KEY = os.getenv("API_KEY", "sk-vRu126d626325944f7040b39845200bafd41123d8f3g48Ol")
    API_BASE = os.getenv("API_BASE", "https://api.gptsapi.net/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo-0125")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))

def create_chat_model():
    return ChatOpenAI(
        model_name=APIConfig.MODEL_NAME,
        openai_api_key=APIConfig.API_KEY,
        openai_api_base=APIConfig.API_BASE,
        temperature=APIConfig.TEMPERATURE,
        max_tokens=APIConfig.MAX_TOKENS,
    )

chat = create_chat_model()

summarize_prompt = ChatPromptTemplate.from_template(
    "Summarize the following text in no more than 2 sentences: {question}"
)
answer_prompt = ChatPromptTemplate.from_template(
    "Answer the following question: {question}"
)
summarize_chain = summarize_prompt | chat | StrOutputParser()
answer_chain = answer_prompt | chat | StrOutputParser()

@lru_cache(maxsize=100)
def get_response(question: str, summarize_chain: Optional[RunnableSequence] = None, answer_chain: Optional[RunnableSequence] = None) -> str:
    summarize_chain = summarize_chain or globals()['summarize_chain']
    answer_chain = answer_chain or globals()['answer_chain']
    try:
        if any(keyword in question.lower() for keyword in ['summarize', 'summarise', 'summary']):
            return summarize_chain.invoke({"question": question})
        else:
            return answer_chain.invoke({"question": question})
    except OpenAIError as e:
        print(f"API error: {str(e)}")
        return "抱歉，处理您的请求时出现了错误。"

class LangChainAgent:
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or create_chat_model()
        self.qa_template = PromptTemplate(
            template="Question: {question}\nAnswer: Let's approach this step-by-step:",
            input_variables=["question"],
        )
        self.summarize_template = PromptTemplate(
            template="Summarize the following text in no more than 2 sentences: \n{text}\nSummary: ",
            input_variables=["text"],
        )

    def _run_chain(self, chain: RunnableSequence, inputs: Dict[str, Any]) -> str:
        try:
            response = chain.invoke(inputs)
            return response if isinstance(response, str) else str(response)
        except OpenAIError as e:
            print(f"API error: {str(e)}")
            return "Sorry, an error occurred while processing your request."
        
    def run_qa_task(self, query: str) -> str:
        try:
            chain = self.qa_template | self.llm | StrOutputParser()
            return self._run_chain(chain, {"question": query})
        except OpenAIError as e:
            print(f"API error in run_qa_task: {str(e)}")
            return "Sorry, an error occurred while processing your request."

    def run_summarization_task(self, text: str) -> str:
        try:
            chain = self.summarize_template | self.llm | StrOutputParser()
            return self._run_chain(chain, {"text": text})
        except OpenAIError as e:
            print(f"API error in run_summarization_task: {str(e)}")
            return "Sorry, an error occurred while summarizing the text."
        
    def run_generation_task(self, prompt: str) -> str:
        try:
            response = self.llm(prompt)
            if isinstance(response, str):
                return response
            elif hasattr(response, "content"):
                return response.content
            else:
                return str(response)
        except Exception as e:
            print(f"Error in run_generation_task: {str(e)}")
            return "Sorry, unable to generate text."

# 增加新的方法以支持未来扩展
    def analyze_sentiment(self, text: str) -> str:
        sentiment_template = PromptTemplate(
            template="Analyze the sentiment of the following text: {text}\nSentiment:",
            input_variables=["text"],
        )
        chain = sentiment_template | self.llm | StrOutputParser()
        return self._run_chain(chain, {"text": text})

    def extract_keywords(self, text: str) -> str:
        keyword_template = PromptTemplate(
            template="Extract the main keywords from the following text: {text}\nKeywords:",
            input_variables=["text"],
        )
        chain = keyword_template | self.llm | StrOutputParser()
        return self._run_chain(chain, {"text": text})