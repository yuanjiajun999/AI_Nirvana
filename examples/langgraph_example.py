# examples/langgraph_example.py

from src.core.langgraph import LangGraph

def main():
    lang_graph = LangGraph()

    # 知识检索示例
    query = "Who invented the telephone?"
    result = lang_graph.retrieve_knowledge(query)
    print(f"Query: {query}")
    print(f"Retrieved knowledge: {result}\n")

    # 推理示例
    premise = "All birds can fly"
    conclusion = "Penguins can fly"
    result = lang_graph.reason(premise, conclusion)
    print(f"Premise: {premise}")
    print(f"Conclusion: {conclusion}")
    print(f"Reasoning result: {result}\n")

    # 常识推理示例
    context = "It's raining outside and John doesn't have an umbrella"
    result = lang_graph.infer_commonsense(context)
    print(f"Context: {context}")
    print(f"Commonsense inference: {result}")

if __name__ == "__main__":
    main()