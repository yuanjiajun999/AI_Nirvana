from typing import Any, Dict


class KnowledgeBase:
    def __init__(self):
        self.knowledge: Dict[str, Any] = {}

    def add_knowledge(self, key: str, value: Any) -> None:
        """添加知识到知识库"""
        self.knowledge[key] = value

    def get_knowledge(self, key: str) -> Any:
        """从知识库获取知识"""
        return self.knowledge.get(key)

    def update_knowledge(self, key: str, value: Any) -> None:
        """更新知识库中的知识"""
        if key in self.knowledge:
            self.knowledge[key] = value
        else:
            raise KeyError(f"知识 '{key}' 不存在于知识库中")

    def delete_knowledge(self, key: str) -> None:
        """从知识库删除知识"""
        if key in self.knowledge:
            del self.knowledge[key]
        else:
            raise KeyError(f"知识 '{key}' 不存在于知识库中")

    def list_all_knowledge(self) -> Dict[str, Any]:
        """列出知识库中的所有知识"""
        return self.knowledge
