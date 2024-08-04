import pandas as pd
import numpy as np
from auto_feature_engineer import AutoFeatureEngineer

# 创建一个示例数据集
np.random.seed(0)
data = pd.DataFrame({
    'id': range(1000),
    'feature1': np.random.rand(1000),
    'feature2': np.random.choice(['A', 'B', 'C'], 1000),
    'date': pd.date_range(start='2021-01-01', periods=1000),
    'target': np.random.randint(0, 2, 1000)
})

# 初始化 AutoFeatureEngineer
afe = AutoFeatureEngineer(data, target_column='target')

# 创建实体集
es = afe.create_entity_set(index_column='id', time_index='date')

# 生成特征
feature_matrix, feature_defs = afe.generate_features(max_depth=2)

# 获取重要特征
important_features = afe.get_important_features(n=5, method='correlation')
print("Top 5 important features:", important_features)

# 获取特征类型
feature_types = afe.get_feature_types()
print("Feature types:", feature_types)

# 移除低信息量特征
removed_features = afe.remove_low_information_features(threshold=0.95)
print("Removed low information features:", removed_features)

# 移除高相关性特征
removed_correlated = afe.remove_highly_correlated_features(threshold=0.9)
print("Removed highly correlated features:", removed_correlated)

# 标准化特征
afe.normalize_features(method='standard')

# 编码分类特征
afe.encode_categorical_features(method='onehot')

# 获取最终的特征矩阵
final_feature_matrix = afe.get_feature_matrix()
print("Final feature matrix shape:", final_feature_matrix.shape)