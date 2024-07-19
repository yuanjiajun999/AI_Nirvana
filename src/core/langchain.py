from typing import Dict, Any  
from langchain_openai import ChatOpenAI  
from langchain.prompts import PromptTemplate, ChatPromptTemplate  
from langchain_core.runnables import RunnableSequence, RunnablePassthrough  
from langchain_core.output_parsers import StrOutputParser  
import os  
from dotenv import load_dotenv  
from openai import OpenAIError  
from functools import lru_cache  

load_dotenv()  

class APIConfig:  
    API_KEY = os.getenv("API_KEY", "sk-vRu126d626325944f7040b39845200bafd41123d8f3g48Ol")  
    API_BASE = os.getenv("API_BASE", "https://api.gptsapi.net/v1")  
    MODEL_NAME = "gpt-3.5-turbo-0125"  
    TEMPERATURE = 0.7  
    MAX_TOKENS = 256  

chat = ChatOpenAI(  
    model_name=APIConfig.MODEL_NAME,  
    openai_api_key=APIConfig.API_KEY,  
    openai_api_base=APIConfig.API_BASE,  
    temperature=APIConfig.TEMPERATURE,  
    max_tokens=APIConfig.MAX_TOKENS  
)  

# 创建全局的 chain  
summarize_prompt = ChatPromptTemplate.from_template("Summarize the following text in no more than 2 sentences: {question}")  
answer_prompt = ChatPromptTemplate.from_template("Answer the following question: {question}")  
summarize_chain = summarize_prompt | chat | StrOutputParser()  
answer_chain = answer_prompt | chat | StrOutputParser()  

@lru_cache(maxsize=100)  
def get_response(question: str) -> str:  
    try:  
        if "summarize" in question.lower():  
            return summarize_chain.invoke({"question": question})  
        else:  
            return answer_chain.invoke({"question": question})  
    except OpenAIError as e:  
        print(f"API 错误: {str(e)}")  
        return "抱歉，处理您的请求时出现了错误。"  

class LangChainAgent:  
    def __init__(self):  
        self.llm = chat  
        self.qa_template = PromptTemplate(  
            template="Question: {question}\nAnswer: Let's approach this step-by-step:",  
            input_variables=["question"]  
        )  
        self.summarize_template = PromptTemplate(  
            template="Summarize the following text in no more than 2 sentences:\n{text}\nSummary:",  
            input_variables=["text"]  
        )  

    def _run_chain(self, chain: RunnableSequence, inputs: Dict[str, Any]) -> str:  
        try:  
            response = chain.invoke(inputs)  
            return response if isinstance(response, str) else str(response)  
        except OpenAIError as e:  
            print(f"API 错误: {str(e)}")  
            return "抱歉，处理您的请求时出现了错误。"  

    def run_qa_task(self, query: str) -> str:  
        chain = self.qa_template | self.llm | StrOutputParser()  
        return self._run_chain(chain, {"question": query})  

    def run_summarization_task(self, text: str) -> str:  
        chain = self.summarize_template | self.llm | StrOutputParser()  
        return self._run_chain(chain, {"text": text})  

    def run_generation_task(self, prompt: str) -> str:  
        return self._run_chain(self.llm | StrOutputParser(), {"text": prompt})