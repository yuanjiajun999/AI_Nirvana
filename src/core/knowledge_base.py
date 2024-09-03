import json
import logging 
from typing import Dict, Any, List
import threading
import logging
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')  
logger = logging.getLogger(__name__) 

class KnowledgeBase:  
    def __init__(self, config, api_client, file_path: str = "knowledge_base.json"):  
        self.config = config  
        self.api_client = api_client  
        self.file_path = file_path  
        self.knowledge: Dict[str, Any] = {}  
        self.lock = threading.Lock()  
        self.load_knowledge()  
        
        logging.basicConfig(filename='knowledge_base.log', level=logging.INFO)  
        
        # 使用配置中的 API 密钥和基础 URL  
        self.openai_api_key = config.api_key  
        self.openai_api_base = config.api_base  
        
        logging.info("Starting to create embeddings...")  
        self.embeddings = OpenAIEmbeddings(  
            openai_api_key=self.openai_api_key,  
            openai_api_base=self.openai_api_base, 
            timeout=60  # 设置60秒超时 
        )  
        logging.info("Embeddings created successfully.")  
        
        logging.info("Starting to initialize FAISS vector store...")  
        self.vector_store = FAISS.from_texts(["Initial knowledge base"], embedding=self.embeddings)  
        logging.info("FAISS vector store initialized successfully.")  
        
        logging.info("Initializing ChatOpenAI...")  
        self.llm = ChatOpenAI(  
            temperature=0,  
            openai_api_key=self.openai_api_key,  
            openai_api_base=self.openai_api_base  
        )  
        logging.info("ChatOpenAI initialized successfully.")  
        
        logging.info("Setting up RetrievalQA chain...")  
        self.qa_chain = RetrievalQA.from_chain_type(  
            llm=self.llm,  
            chain_type="stuff",  
            retriever=self.vector_store.as_retriever()  
        )  
        logging.info("RetrievalQA chain set up successfully.")  
        
        logging.info("KnowledgeBase initialization completed.")  
        
    def add_knowledge(self, key: str, value: Any) -> dict:
        """添加知识到知识库"""
        if not key or not isinstance(key, str):
            logger.error("Invalid key provided for adding knowledge.")
            return {"message": "Invalid key provided", "success": False}
    
        if not value:  # 检查值是否为空
            logger.error("Invalid value provided for adding knowledge.")
            return {"message": "Invalid value provided", "success": False}
    
        try:
            logger.debug(f"Attempting to add knowledge with key: {key}")
            logger.debug(f"Current keys in knowledge base: {list(self.knowledge.keys())}")
        
            if not key.isalnum():  # 检查键是否只包含字母和数字
                logger.warning(f"Key '{key}' contains special characters.")
                user_input = input("警告：键应该只包含字母和数字。是否继续？(y/n): ").strip().lower()
                if user_input != 'y':
                    logger.info(f"Operation cancelled for key: {key}")
                    return {"message": "操作已取消。", "success": False}
        
            if key in self.knowledge:
                logger.debug(f"Key '{key}' already exists. Prompting for overwrite.")
                user_input = input(f"键 '{key}' 已存在。是否要覆盖？(y/n): ").strip().lower()
                logger.debug(f"User input for overwrite: {user_input}")
                if user_input != 'y':
                    logger.info(f"Operation cancelled for key: {key}")
                    return {"message": "操作已取消。", "success": False}
        
            logger.debug(f"Adding/updating knowledge for key: {key}")
            self.knowledge[key] = value
            logger.info(f"Knowledge added/updated: {key}")
            return {"message": f"Successfully added/updated knowledge: {key}", "success": True}
        except Exception as e:
            logger.error(f"Error adding/updating knowledge: {str(e)}", exc_info=True)
            return {"message": f"Failed to add/update knowledge: {str(e)}", "success": False}

    def load_knowledge(self):
        if os.path.exists(self.file_path):
            with self.lock:
                with open(self.file_path, 'r') as f:
                    self.knowledge = json.load(f)
        else:
            self.knowledge = {}

    def save_knowledge(self):
        with self.lock:
            with open(self.file_path, 'w') as f:
                json.dump(self.knowledge, f, indent=2)

    def get(self, key: str) -> Any:
        logger.info(f"KnowledgeBase: Getting key {key}") 
        with self.lock:
            return self.knowledge.get(key)  

    def set(self, key: str, value: Any):
        logger.info(f"KnowledgeBase: Setting key {key}") 
        with self.lock:
            self.knowledge[key] = value
            self.save_knowledge()
            self.vector_store.add_texts([value['content']], metadatas=[{"title": key}])
        logging.info(f"Added/Updated key: {key} at {datetime.now()}")

    def delete(self, key: str):
        logging.info(f"KnowledgeBase: Deleting key {key}")
        with self.lock:
            if key in self.knowledge:
                del self.knowledge[key]
                self.save_knowledge()
                logging.info(f"Deleted key: {key} at {datetime.now()}")

    def search(self, query: str) -> List[str]:
        logging.info(f"KnowledgeBase: Searching for query: {query}")  
        docs = self.vector_store.similarity_search(query)
        return [doc.metadata.get('title', '') for doc in docs]

    def get_all_keys(self) -> List[str]:
        logging.info("KnowledgeBase: Getting all keys") 
        with self.lock:
            return list(self.knowledge.keys())

    def query(self, question: str) -> str:  
        logging.info(f"KnowledgeBase: Querying: {question}")  
        result = self.qa_chain.run(question)  
        logging.info(f"KnowledgeBase: Query result: {result}")  
        return result

    def fetch_wikipedia_content(self, topic: str) -> tuple[str, str]:
        logging.info(f"KnowledgeBase: Fetching Wikipedia content for topic: {topic}")  
        url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.find(id="firstHeading").text
            content = ""
            for p in soup.find(id="mw-content-text").find_all("p", class_=""):
                content += p.text + "\n"
            return title, content.strip()
        return None, None

    def add_wikipedia_entry(self, topic: str):
        logging.info(f"KnowledgeBase: Adding Wikipedia entry for topic: {topic}")  
        title, content = self.fetch_wikipedia_content(topic)
        if title and content:
            self.set(title, {"content": content, "source": f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"})
            print(f"Added entry: {title}")
        else:
            print(f"Failed to fetch information about {topic}")
            
    def test_operation(self):  
        logging.info("KnowledgeBase: Starting test operation")  
        test_key = "test_entry"  
        test_value = {"content": "This is a test entry", "source": "test"}  
        self.set(test_key, test_value)  
        retrieved_value = self.get(test_key)  
        assert retrieved_value == test_value, "Test operation failed: set/get mismatch"  
        self.delete(test_key)  
        assert self.get(test_key) is None, "Test operation failed: delete failed"  
        return "Knowledge base test operation successful"

    def update(self, key: str, value: str) -> dict:
        if key in self.knowledge:
            old_value = self.knowledge[key]
            self.knowledge[key] = value
            return {"message": f"已更新知识：{key}", "old_value": old_value, "new_value": value}
        else:
            return {"message": f"未找到知识：{key}"}

    def list_all(self) -> List[str]:
        return list(self.knowledge.keys())    
        
class KnowledgeBaseManager:
    def __init__(self):
        self.kb = KnowledgeBase()

    def query(self, key: str) -> Dict[str, Any]:
        result = self.kb.get(key)
        if result:
            return {"found": True, "data": result}
        else:
            return {"found": False, "data": None}

    def add_or_update(self, key: str, value: Any) -> Dict[str, bool]:
        try:
            self.kb.set(key, value)
            return {"success": True}
        except Exception as e:
            logging.error(f"Error adding/updating key {key}: {str(e)}")
            return {"success": False, "error": str(e)}

    def remove(self, key: str) -> Dict[str, bool]:
        try:
            self.kb.delete(key)
            return {"success": True}
        except Exception as e:
            logging.error(f"Error removing key {key}: {str(e)}")
            return {"success": False, "error": str(e)}

    def search_knowledge(self, query: str) -> List[str]:
        return self.kb.search(query)

    def get_all_knowledge_keys(self) -> List[str]:
        return self.kb.get_all_keys()

    def ask_question(self, question: str) -> str:
        return self.kb.query(question)

    def add_wikipedia_entry(self, topic: str) -> Dict[str, bool]:
        try:
            self.kb.add_wikipedia_entry(topic)
            return {"success": True}
        except Exception as e:
            logging.error(f"Error adding Wikipedia entry for {topic}: {str(e)}")
            return {"success": False, "error": str(e)}

