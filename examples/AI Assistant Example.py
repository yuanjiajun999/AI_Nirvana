import logging  
from src.core.language_model import LanguageModel  
from src.core.knowledge_base import KnowledgeBase  
from src.core.reasoning import ReasoningEngine  
from src.core.security import SecurityManager  
from src.core.multimodal import MultimodalInterface  

class AIAssistant:  
    def __init__(self, model_name="gpt-3.5-turbo"):  
        self.language_model = LanguageModel(model_name)  
        self.knowledge_base = KnowledgeBase()  
        self.reasoning_engine = ReasoningEngine()  
        self.security_manager = SecurityManager()  
        self.multimodal_interface = MultimodalInterface()  
        self.context = []  
        logging.info(f"AI Assistant initialized with model: {model_name}")  

    def process_input(self, user_input):  
        logging.info(f"User input: {user_input}")  
        
        # 使用多模态接口处理输入  
        processed_input = self.multimodal_interface.process(user_input)  
        
        # 检查输入安全性  
        if not self.security_manager.is_safe(processed_input):  
            return "I'm sorry, but I can't process that input due to safety concerns."  
        
        # 从知识库检索相关信息  
        relevant_info = self.knowledge_base.retrieve(processed_input)  
        
        # 使用推理引擎进行分析  
        reasoning_result = self.reasoning_engine.reason(processed_input, relevant_info)  
        
        # 生成响应  
        response = self.language_model.generate_response(reasoning_result)  
        
        logging.info(f"System response: {response}")  
        return response  

    def change_model(self, new_model):  
        self.language_model = LanguageModel(new_model)  
        logging.info(f"Model changed to: {new_model}")  

    def clear_context(self):  
        self.context = []  
        logging.info("Context cleared.")  

    def analyze_sentiment(self, text):  
        logging.info(f"Sentiment analysis request: {text}")  
        # 假设有一个情感分析方法  
        sentiment = self.language_model.analyze_sentiment(text)  
        logging.info(f"Sentiment analysis result: {sentiment}")  
        return sentiment  

    def encrypt_data(self, data):  
        encrypted = self.security_manager.encrypt(data)  
        logging.info("Data encrypted successfully")  
        return encrypted  

    def decrypt_data(self, encrypted_data):  
        decrypted = self.security_manager.decrypt(encrypted_data)  
        logging.info("Data decrypted successfully")  
        return decrypted  

    def execute_code(self, code):  
        logging.info(f"Code execution request: {code}")  
        # 假设有一个安全的代码执行环境  
        result = self.security_manager.safe_execute(code)  
        logging.info(f"Code execution result: {result}")  
        return result  

    def get_available_models(self):  
        models = self.language_model.list_models()  
        logging.info(f"Available models: {models}")  
        return models  

    def plan_task(self, task_description):  
        logging.info(f"Task planning request: {task_description}")  
        plan = self.reasoning_engine.generate_plan(task_description)  
        logging.info(f"Generated plan: {plan}")  
        return plan  

    def summarize_text(self, long_text):  
        logging.info(f"Summarization request: {long_text[:50]}...")  
        summary = self.language_model.summarize(long_text)  
        logging.info(f"Summary: {summary}")  
        return summary