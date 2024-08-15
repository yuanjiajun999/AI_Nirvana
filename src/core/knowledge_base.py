from typing import Any, Dict, List
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBase:
    def __init__(self):
        self.knowledge: Dict[str, Any] = {}

    def add_knowledge(self, key: str, value: Any) -> None:
        """添加知识到知识库"""
        if not key or not isinstance(key, str):
            logger.error("Invalid key provided for adding knowledge.")
            raise ValueError("Key must be a non-empty string.")
        self.knowledge[key] = value
        logger.info(f"Knowledge added: {key}")

    def get_knowledge(self, key: str) -> Any:
        """从知识库获取知识"""
        if key not in self.knowledge:
            logger.warning(f"Knowledge with key '{key}' not found.")
            raise KeyError(f"知识 '{key}' 不存在于知识库中")
        return self.knowledge[key]

    def update_knowledge(self, key: str, value: Any) -> None:
        """更新知识库中的知识"""
        if key not in self.knowledge:
            logger.warning(f"Cannot update non-existing knowledge with key '{key}'.")
            raise KeyError(f"知识 '{key}' 不存在于知识库中")
        self.knowledge[key] = value
        logger.info(f"Knowledge updated: {key}")

    def delete_knowledge(self, key: str) -> None:
        """从知识库删除知识"""
        if key not in self.knowledge:
            logger.warning(f"Cannot delete non-existing knowledge with key '{key}'.")
            raise KeyError(f"知识 '{key}' 不存在于知识库中")
        del self.knowledge[key]
        logger.info(f"Knowledge deleted: {key}")

    def list_all_knowledge(self) -> Dict[str, Any]:
        """列出知识库中的所有知识"""
        return self.knowledge

    def retrieve(self, query: str) -> List[Any]:
        """根据查询检索相关知识"""
        relevant_knowledge = []
        query = query.lower()
        for key, value in self.knowledge.items():
            if query in key.lower() or (isinstance(value, str) and query in value.lower()):
                relevant_knowledge.append(value)
        logger.info(f"Retrieved knowledge for query '{query}': {relevant_knowledge}")
        return relevant_knowledge
