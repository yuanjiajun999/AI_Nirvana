from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_community.graphs import NetworkxEntityGraph
from langchain.chains import GraphQAChain
from langchain_core.runnables import RunnableSequence
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from openai import OpenAIError
from functools import lru_cache

load_dotenv()

class APIConfig:
    MODEL_NAME = "gpt-3.5-turbo-0125"
    API_KEY = os.getenv("API_KEY")
    API_BASE = "https://api.gptsapi.net/v1"
    TEMPERATURE = 0.7
    MAX_TOKENS = 256

class LangGraph:
    def __init__(self):
        try:
            self.llm = ChatOpenAI(
                model_name=APIConfig.MODEL_NAME,
                openai_api_key=APIConfig.API_KEY,
                openai_api_base=APIConfig.API_BASE,
                temperature=APIConfig.TEMPERATURE,
                max_tokens=APIConfig.MAX_TOKENS
            )
            self.graph = NetworkxEntityGraph()
            
            self.entity_extraction_prompt = PromptTemplate(
                template="Extract entities from the following text:\n\n{text}\n\nEntities:",
                input_variables=["text"]
            )
            self.entity_extraction_chain = self.entity_extraction_prompt | self.llm
            
            self.qa_chain = GraphQAChain.from_llm(
                llm=self.llm,
                graph=self.graph,
                verbose=True
            )
        except OpenAIError as e:
            print(f"API 错误: {str(e)}")
            raise

    @lru_cache(maxsize=100)
    def _cached_run(self, query: str) -> str:
        try:
            return self.qa_chain.invoke(query)
        except OpenAIError as e:
            print(f"API 错误: {str(e)}")
            return "抱歉，处理您的请求时出现了错误。"

    def retrieve_knowledge(self, query: str) -> str:
        return self._cached_run(query)

    def reason(self, premise: str, conclusion: str) -> str:
        query = f"Given the premise '{premise}', is the conclusion '{conclusion}' valid?"
        return self._cached_run(query)

    def infer_commonsense(self, context: str) -> str:
        query = f"Based on the context '{context}', what can we infer?"
        return self._cached_run(query)