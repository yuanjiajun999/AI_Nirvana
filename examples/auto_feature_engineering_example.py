import pandas as pd  
import numpy as np  
from auto_feature_engineering import AutoFeatureEngineer  

# 创建示例数据  
np.random.seed(0)  
data = pd.DataFrame({  
    'id': range(1000),  
    'age': np.random.randint(18, 80, 1000),  
    'income': np.random.randint(20000, 100000, 1000),  
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),  
    'employment': np.random.choice(['Employed', 'Unemployed', 'Self-employed'], 1000),  
    'loan_amount': np.random.randint(5000, 50000, 1000),  
    'loan_term': np.random.choice([12, 24, 36, 48, 60], 1000),  
    'loan_status': np.random.choice(['Approved', 'Rejected'], 1000)  
})  

# 初始化 AutoFeatureEngineer  
afe = AutoFeatureEngineer(data, target_column='loan_status')  

# 创建实体集  
es = afe.create_entity_set(index_column='id')  

# 生成特征  
feature_matrix, feature_defs = afe.generate_features(max_depth=2)  

# 获取重要特征  
important_features = afe.get_important_features(n=5, method='mutual_info')  
print("Top 5 important features:", important_features)  

# 获取特征类型  
feature_types = afe.get_feature_types()  
print("Feature types:", feature_types)  

# 移除低信息特征  
removed_features = afe.remove_low_information_features(threshold=0.95)  
print("Removed low information features:", removed_features)  

# 移除高度相关特征  
removed_correlated = afe.remove_highly_correlated_features(threshold=0.9)  
print("Removed highly correlated features:", removed_correlated)  

# 归一化特征  
afe.normalize_features(method='standard')  

# 编码分类特征  
afe.encode_categorical_features(method='onehot')  

# 获取最终的特征矩阵  
final_feature_matrix = afe.get_feature_matrix()  
print("Final feature matrix shape:", final_feature_matrix.shape)