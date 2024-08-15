from src.core.langgraph import LangGraph

def main():
    # 初始化 LangGraph
    lang_graph = LangGraph()

    # 添加实体
    lang_graph.add_entity("Python", {"type": "programming_language", "paradigm": "multi-paradigm"})
    lang_graph.add_entity("Java", {"type": "programming_language", "paradigm": "object-oriented"})

    # 添加关系
    lang_graph.add_relationship("Python", "Java", "is_different_from")

    # 检索知识
    knowledge = lang_graph.retrieve_knowledge("What is Python?")
    print("Retrieved knowledge:", knowledge)

    # 提取实体
    text = "Python and Java are both popular programming languages."
    entities = lang_graph.extract_entities(text)
    print("Extracted entities:", entities)

    # 进行推理
    reason_result = lang_graph.reason("Python is a programming language", "Python can be used for web development")
    print("Reasoning result:", reason_result)

    # 常识推理
    inference_result = lang_graph.infer_commonsense("A programmer is working on a Python project")
    print("Inference result:", inference_result)

    # 语义搜索
    search_results = lang_graph.semantic_search("object-oriented programming")
    print("Semantic search results:", search_results)

    # 获取图的摘要
    graph_summary = lang_graph.get_graph_summary()
    print("Graph summary:", graph_summary)

    # 导出图
    export_result = lang_graph.export_graph()
    print(export_result)

if __name__ == "__main__":
    main()