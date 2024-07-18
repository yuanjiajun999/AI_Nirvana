from langchain_community.chat_models import ChatOpenAI  
from langchain.chains import LLMChain  
from langchain.prompts import PromptTemplate  
import os  
from dotenv import load_dotenv  
from openai import OpenAIError  

load_dotenv()  

class LangChainAgent:  
    def __init__(self):  
        try:  
            self.llm = ChatOpenAI(  
                model_name="gpt-3.5-turbo-0125",  
                openai_api_key=os.getenv("API_KEY"),  
                openai_api_base="https://api.gptsapi.net/v1",  
                temperature=0.7,  
                max_tokens=256  
            )  
            template = "Question: {question}\nAnswer: Let's approach this step-by-step:"  
            prompt = PromptTemplate(template=template, input_variables=["question"])  
            self.qa_chain = LLMChain(llm=self.llm, prompt=prompt)  
        except OpenAIError as e:  
            print(f"API 错误: {str(e)}")  
            # 根据需要进行适当的错误处理，例如设置默认值或抛出异常  

    def run_qa_task(self, query):  
        try:  
            return self.qa_chain.run(question=query)  
        except OpenAIError as e:  
            print(f"API 错误: {str(e)}")  
            return "抱歉，处理您的请求时出现了错误。"  

    def run_summarization_task(self, text):  
        try:  
            template = "Summarize the following text:\n{text}\nSummary:"  
            prompt = PromptTemplate(template=template, input_variables=["text"])  
            chain = LLMChain(llm=self.llm, prompt=prompt)  
            return chain.run(text=text)  
        except OpenAIError as e:  
            print(f"API 错误: {str(e)}")  
            return "抱歉，无法生成摘要。"  

    def run_generation_task(self, prompt):  
        try:  
            return self.llm.predict(prompt)  
        except OpenAIError as e:  
            print(f"API 错误: {str(e)}")  
            return "抱歉，无法生成文本。"  