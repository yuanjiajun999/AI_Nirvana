from langchain_community.chat_models import ChatOpenAI  
from langchain_community.graphs import NetworkxEntityGraph  
from langchain.chains import GraphQAChain  
from langchain.chains import LLMChain  
from langchain.prompts import PromptTemplate  
import os  
from dotenv import load_dotenv  
from openai import OpenAIError  

load_dotenv()  

class LangGraph:  
    def __init__(self):  
        try:  
            self.llm = ChatOpenAI(  
                model_name="gpt-3.5-turbo-0125",  # 使用 WildCard 支持的模型  
                openai_api_key=os.getenv("API_KEY"),  
                openai_api_base="https://api.gptsapi.net/v1",  
                temperature=0.7,  
                max_tokens=256  
            )  
            self.graph = NetworkxEntityGraph()  
            
            entity_extraction_template = "Extract entities from the following text:\n\n{text}\n\nEntities:"  
            entity_extraction_prompt = PromptTemplate(template=entity_extraction_template, input_variables=["text"])  
            self.entity_extraction_chain = LLMChain(llm=self.llm, prompt=entity_extraction_prompt)  
            
            self.qa_chain = GraphQAChain.from_llm(  
                llm=self.llm,  
                graph=self.graph,  
                verbose=True  
            )  
        except OpenAIError as e:  
            print(f"API 错误: {str(e)}")  
            # 根据需要进行适当的错误处理，例如设置默认值或抛出异常  

    def retrieve_knowledge(self, query):  
        try:  
            return self.qa_chain.run(query)  
        except OpenAIError as e:  
            print(f"API 错误: {str(e)}")  
            return "抱歉，无法检索知识。"  

    def reason(self, premise, conclusion):  
        try:  
            query = f"Given the premise '{premise}', is the conclusion '{conclusion}' valid?"  
            return self.qa_chain.run(query)  
        except OpenAIError as e:  
            print(f"API 错误: {str(e)}")  
            return "抱歉，无法进行推理。"  

    def infer_commonsense(self, context):  
        try:  
            query = f"Based on the context '{context}', what can we infer?"  
            return self.qa_chain.run(query)  
        except OpenAIError as e:  
            print(f"API 错误: {str(e)}")  
            return "抱歉，无法进行常识推理。"