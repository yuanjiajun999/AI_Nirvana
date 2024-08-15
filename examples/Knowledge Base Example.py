from src.core.knowledge_base import KnowledgeBase

def main():
    # 初始化知识库
    kb = KnowledgeBase()
    
    # 添加知识
    kb.add_knowledge("AI", "Artificial Intelligence refers to the simulation of human intelligence in machines.")
    kb.add_knowledge("ML", "Machine Learning is a subset of AI that enables systems to learn from data.")

    # 获取知识
    try:
        ai_knowledge = kb.get_knowledge("AI")
        print(f"AI Knowledge: {ai_knowledge}")
    except KeyError as e:
        print(e)

    # 更新知识
    try:
        kb.update_knowledge("AI", "AI involves the creation of intelligent agents.")
        updated_ai_knowledge = kb.get_knowledge("AI")
        print(f"Updated AI Knowledge: {updated_ai_knowledge}")
    except KeyError as e:
        print(e)

    # 列出所有知识
    all_knowledge = kb.list_all_knowledge()
    print("All Knowledge:")
    for key, value in all_knowledge.items():
        print(f"{key}: {value}")

    # 检索知识
    search_results = kb.retrieve("intelligence")
    print("Search Results for 'intelligence':")
    for result in search_results:
        print(result)

    # 删除知识
    try:
        kb.delete_knowledge("ML")
        print("ML knowledge deleted successfully.")
    except KeyError as e:
        print(e)

    # 再次列出所有知识
    all_knowledge = kb.list_all_knowledge()
    print("All Knowledge after deletion:")
    for key, value in all_knowledge.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
