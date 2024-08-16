# AutoFeatureEngineer 类文档

## 简介

AutoFeatureEngineer 是一个强大的自动特征工程类，它利用 Featuretools 库来生成、选择和处理特征。这个类提供了一套全面的工具，用于数据准备、特征生成和特征选择，可以显著提升机器学习模型的性能。

## 类初始化

```python
afe = AutoFeatureEngineer(data, target_column)
```

- `data`: pandas DataFrame，包含原始数据
- `target_column`: 字符串，目标列的名称

## 主要方法

### 1. create_entity_set(index_column, time_index=None)

创建一个 Featuretools EntitySet。

**参数**:
- `index_column`: 字符串，用作索引的列名
- `time_index`: 可选，字符串，用作时间索引的列名

**返回**: Featuretools EntitySet

**使用示例**:
```python
es = afe.create_entity_set(index_column='id')
```

### 2. generate_features(max_depth=2, primitives=None, show_warnings=False)

使用 Featuretools 的深度特征综合生成特征。

**参数**:
- `max_depth`: 整数，特征生成的最大深度
- `primitives`: 可选，列表，要使用的原始函数列表
- `show_warnings`: 布尔值，是否显示警告

**返回**: (特征矩阵, 特征定义) 的元组

**使用示例**:
```python
feature_matrix, feature_defs = afe.generate_features(max_depth=2)
```

### 3. get_important_features(n=10, method='correlation')

获取最重要的特征。

**参数**:
- `n`: 整数，返回的重要特征数量
- `method`: 字符串，用于确定特征重要性的方法 ('correlation', 'mutual_info', 'mutual_info_regression', 或 'mutual_info_classif')

**返回**: 重要特征名称的列表

**使用示例**:
```python
important_features = afe.get_important_features(n=5, method='mutual_info')
```

### 4. remove_low_information_features(threshold=0.95)

移除低信息内容的特征。

**参数**:
- `threshold`: 浮点数，确定低信息内容的阈值

**返回**: 被移除的特征名称列表

**使用示例**:
```python
removed_features = afe.remove_low_information_features(threshold=0.9)
```

### 5. remove_highly_correlated_features(threshold=0.9)

移除高度相关的特征。

**参数**:
- `threshold`: 浮点数，用于移除特征的相关阈值

**返回**: 被移除的特征名称列表

**使用示例**:
```python
removed_correlated = afe.remove_highly_correlated_features(threshold=0.8)
```

### 6. normalize_features(method='standard')

归一化特征矩阵中的数值特征。

**参数**:
- `method`: 字符串，要使用的归一化方法 ('standard' 或 'minmax')

**使用示例**:
```python
afe.normalize_features(method='standard')
```

### 7. encode_categorical_features(method='onehot')

编码特征矩阵中的分类特征。

**参数**:
- `method`: 字符串，要使用的编码方法 ('onehot' 或 'label')

**使用示例**:
```python
afe.encode_categorical_features(method='onehot')
```

### 8. get_feature_types()

获取特征矩阵中所有特征的类型。

**返回**: 将特征名称映射到其类型的字典

**使用示例**:
```python
feature_types = afe.get_feature_types()
```

### 9. get_feature_matrix()

获取特征矩阵。

**返回**: pandas DataFrame，包含生成的特征

**使用示例**:
```python
final_feature_matrix = afe.get_feature_matrix()
```

## 完整使用示例

```python
import pandas as pd
from auto_feature_engineering import AutoFeatureEngineer

# 加载数据
data = pd.read_csv('your_data.csv')

# 初始化 AutoFeatureEngineer
afe = AutoFeatureEngineer(data, target_column='target')

# 创建实体集
es = afe.create_entity_set(index_column='id')

# 生成特征
feature_matrix, feature_defs = afe.generate_features(max_depth=2)

# 获取重要特征
important_features = afe.get_important_features(n=5, method='mutual_info')

# 移除低信息特征
afe.remove_low_information_features()

# 归一化特征
afe.normalize_features()

# 编码分类特征
afe.encode_categorical_features()

# 获取最终的特征矩阵
final_feature_matrix = afe.get_feature_matrix()
```

## 最佳实践

1. 在生成特征后，使用 `get_important_features` 来识别最相关的特征。
2. 使用 `remove_low_information_features` 和 `remove_highly_correlated_features` 来优化特征集。
3. 在模型训练前，确保使用 `normalize_features` 和 `encode_categorical_features`。
4. 根据您的具体需求调整参数，如 `max_depth` 和各种阈值。
5. 记得检查生成的特征的类型和描述，以便更好地理解它们。