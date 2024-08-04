# AutoFeatureEngineer 使用文档

`AutoFeatureEngineer` 是一个强大的自动特征工程工具，它利用 Featuretools 库来生成、选择和处理特征。以下是使用此类的基本步骤和主要功能。

## 初始化

首先，导入类并使用您的数据和目标列名初始化：

```python
from auto_feature_engineer import AutoFeatureEngineer

afe = AutoFeatureEngineer(data, target_column='target')
主要功能

创建实体集
在生成特征之前，需要创建一个实体集：
es = afe.create_entity_set(index_column='id', time_index='date')

生成特征
使用深度特征合成生成新特征：
feature_matrix, feature_defs = afe.generate_features(max_depth=2)

获取重要特征
选择最重要的特征：
important_features = afe.get_important_features(n=5, method='correlation')

获取特征类型
了解每个特征的数据类型：
feature_types = afe.get_feature_types()

移除低信息量特征
删除那些在大多数样本中具有相同值的特征：
removed_features = afe.remove_low_information_features(threshold=0.95)

移除高相关性特征
删除高度相关的特征以减少冗余：
removed_correlated = afe.remove_highly_correlated_features(threshold=0.9)

标准化特征
将数值特征标准化：
afe.normalize_features(method='standard')

编码分类特征
将分类特征转换为数值形式：
afe.encode_categorical_features(method='onehot')

获取最终的特征矩阵
在所有处理完成后，获取最终的特征矩阵：
final_feature_matrix = afe.get_feature_matrix()


注意事项

确保在调用其他方法之前先调用 create_entity_set 和 generate_features。
特征生成和处理可能需要较长时间，特别是对于大型数据集。
某些方法（如 remove_low_information_features 和 remove_highly_correlated_features）会直接修改特征矩阵。
使用 get_feature_matrix() 随时获取当前的特征矩阵。

通过这些步骤，您可以自动化特征工程过程，生成丰富的特征集，并对其进行优化和处理。