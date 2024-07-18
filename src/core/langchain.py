from langchain_openai import ChatOpenAI  
from langchain.prompts import PromptTemplate, ChatPromptTemplate  
from langchain_core.runnables import RunnableSequence, RunnablePassthrough  
import os  
from dotenv import load_dotenv  
from openai import OpenAIError  

load_dotenv()  

api_key = os.getenv("API_KEY", "sk-vRu126d626325944f7040b39845200bafd41123d8f3g48Ol")  
api_base = os.getenv("API_BASE", "https://api.gptsapi.net/v1")  

chat = ChatOpenAI(  
    model_name="gpt-3.5-turbo-0125",  
    openai_api_key=api_key,  
    openai_api_base=api_base,  
    temperature=0.7,  
    max_tokens=256  
)  

def get_response(question: str) -> str:  
    if "summarize" in question.lower():  
        prompt = ChatPromptTemplate.from_template("Summarize the following text in no more than 2 sentences: {question}")  
    else:  
        prompt = ChatPromptTemplate.from_template("Answer the following question: {question}")  
    chain = prompt | chat | RunnablePassthrough()  
    response = chain.invoke({"question": question})  
    return response.content if hasattr(response, 'content') else str(response)  

class LangChainAgent:  
    def __init__(self):  
        try:  
            self.llm = chat  
            template = "Question: {question}\nAnswer: Let's approach this step-by-step:"  
            prompt = PromptTemplate(template=template, input_variables=["question"])  
            self.qa_chain = RunnableSequence(prompt, self.llm)  
        except OpenAIError as e:  
            print(f"API 错误: {str(e)}")  

    def run_qa_task(self, query):  
        try:  
            response = self.qa_chain.invoke({"question": query})  
            return response.content if hasattr(response, 'content') else str(response)  
        except OpenAIError as e:  
            print(f"API 错误: {str(e)}")  
            return "抱歉，处理您的请求时出现了错误。"  

    def run_summarization_task(self, text):  
        try:  
            template = "Summarize the following text in no more than 2 sentences:\n{text}\nSummary:"  
            prompt = PromptTemplate(template=template, input_variables=["text"])  
            chain = RunnableSequence(prompt, self.llm)  
            response = chain.invoke({"text": text})  
            return response.content if hasattr(response, 'content') else str(response)  
        except OpenAIError as e:  
            print(f"API 错误: {str(e)}")  
            return "抱歉，无法生成摘要。"  

    def run_generation_task(self, prompt):  
        try:  
            response = self.llm.invoke(prompt)  
            return response.content if hasattr(response, 'content') else str(response)  
        except OpenAIError as e:  
            print(f"API 错误: {str(e)}")  
            return "抱歉，无法生成文本。"