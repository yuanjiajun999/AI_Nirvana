# KnowledgeBase 类文档

## 类简介

`KnowledgeBase` 类用于存储和管理知识条目。该类提供添加、更新、检索、删除和列出知识的功能。

## 方法

### `add_knowledge(key: str, value: Any) -> None`

- **描述**: 添加一个新的知识条目。
- **参数**:
  - `key`: 知识条目的键（字符串）。
  - `value`: 知识条目的值（任意类型）。
- **异常**: 如果 `key` 为空或不是字符串，将抛出 `ValueError`。

### `get_knowledge(key: str) -> Any`

- **描述**: 根据键获取知识条目。
- **参数**:
  - `key`: 要检索的知识条目的键。
- **返回**: 知识条目的值。
- **异常**: 如果 `key` 不存在，将抛出 `KeyError`。

### `update_knowledge(key: str, value: Any) -> None`

- **描述**: 更新现有的知识条目。
- **参数**:
  - `key`: 要更新的知识条目的键。
  - `value`: 新的知识条目值。
- **异常**: 如果 `key` 不存在，将抛出 `KeyError`。

### `delete_knowledge(key: str) -> None`

- **描述**: 删除指定键的知识条目。
- **参数**:
  - `key`: 要删除的知识条目的键。
- **异常**: 如果 `key` 不存在，将抛出 `KeyError`。

### `list_all_knowledge() -> Dict[str, Any]`

- **描述**: 列出所有存储的知识条目。
- **返回**: 字典，包含所有知识条目。

### `retrieve(query: str) -> List[Any]`

- **描述**: 根据查询字符串检索相关知识条目。
- **参数**:
  - `query`: 查询字符串。
- **返回**: 包含相关知识条目的列表。
