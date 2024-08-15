# LangGraph 类文档

`LangGraph` 是一个结合了语言模型、知识图谱和向量存储的强大工具，用于知识管理、推理和查询。

## 初始化

```python
lang_graph = LangGraph()
初始化时，LangGraph 会设置语言模型、知识图谱、向量存储和对话内存。
主要方法
添加实体
lang_graph.add_entity(entity: str, properties: Dict[str, Any])
向图中添加一个新实体，并同时更新向量存储。
添加关系
lang_graph.add_relationship(entity1: str, entity2: str, relationship: str)
在两个实体之间添加一个关系，并更新向量存储。
检索知识
knowledge = lang_graph.retrieve_knowledge(query: str) -> str
基于给定的查询，从知识图谱中检索相关信息。
提取实体
entities = lang_graph.extract_entities(text: str) -> List[str]
从给定的文本中提取实体。
推理
result = lang_graph.reason(premise: str, conclusion: str) -> str
基于给定的前提和结论进行逻辑推理。
常识推理
result = lang_graph.infer_commonsense(context: str) -> str
基于给定的上下文进行常识推理。
语义搜索
results = lang_graph.semantic_search(query: str, k: int = 5) -> List[Tuple[str, float]]
执行语义搜索，返回最相关的 k 个结果及其相似度分数。
获取图摘要
summary = lang_graph.get_graph_summary() -> Dict[str, Any]
返回图的摘要信息，包括节点数、边数、密度和连通组件数。
导出图
result = lang_graph.export_graph(format: str = "graphml") -> str
将图导出为指定格式（默认为 GraphML）。
更新实体
result = lang_graph.update_entity(entity: str, new_properties: Dict[str, Any])
更新现有实体的属性。
删除实体
result = lang_graph.delete_entity(entity: str)
从图中删除指定的实体。
获取所有实体
entities = lang_graph.get_all_entities() -> List[str]
返回图中所有实体的列表。
获取所有关系
relationships = lang_graph.get_all_relationships() -> List[Tuple[str, str, str]]
返回图中所有关系的列表，每个关系表示为一个三元组 (实体1, 实体2, 关系类型)。
注意事项

该类依赖于外部API（如OpenAI），请确保在使用前正确设置API密钥。
某些操作可能会抛出 OpenAIError，已在相关方法中进行了异常处理。
向量存储和知识图谱的操作是同步的，确保两者的一致性。


这个文档提供了 `LangGraph` 类的主要功能概述，包括每个主要方法的简要说明和使用方式。您可以根据需要进一步扩展或修改这个文档。