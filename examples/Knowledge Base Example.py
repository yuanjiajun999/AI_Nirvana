from src.core.knowledge_base import KnowledgeBase


def main():
    kb = KnowledgeBase()

    # 添加知识
    kb.add_knowledge("capital_of_France", "Paris")
    kb.add_knowledge(
        "planets_in_solar_system",
        ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"],
    )

    # 获取知识
    print("Capital of France:", kb.get_knowledge("capital_of_France"))
    print("Planets in the solar system:", kb.get_knowledge("planets_in_solar_system"))

    # 更新知识
    kb.update_knowledge("capital_of_France", "Paris, the City of Light")
    print("Updated capital of France:", kb.get_knowledge("capital_of_France"))

    # 删除知识
    kb.delete_knowledge("planets_in_solar_system")
    print("Planets after deletion:", kb.get_knowledge("planets_in_solar_system"))

    # 列出所有知识
    print("\nAll knowledge in the knowledge base:")
    all_knowledge = kb.list_all_knowledge()
    for key, value in all_knowledge.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
