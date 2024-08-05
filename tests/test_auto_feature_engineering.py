import warnings  
import pytest  
import pandas as pd  
import numpy as np  
import featuretools as ft  
import sys  
import os  
import featuretools as ft
print(f"Featuretools version: {ft.__version__}")

# 添加这一行，将 src 目录添加到 sys.path 中  
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))  

from core.auto_feature_engineering import AutoFeatureEngineer  

# 忽略警告  
warnings.filterwarnings("ignore", category=DeprecationWarning)  
warnings.filterwarnings("ignore", category=FutureWarning)  
warnings.filterwarnings("ignore", category=UserWarning)  
warnings.filterwarnings("ignore", category=ft.exceptions.UnusedPrimitiveWarning)  

@pytest.fixture  
def sample_data():  
    return pd.DataFrame({  
        'id': range(1, 101),  
        'feature1': np.random.rand(100),  
        'feature2': np.random.choice(['A', 'B', 'C'], 100),  
        'feature3': pd.date_range(start='2021-01-01', periods=100, freq='D'),  
        'target': np.random.randint(0, 2, 100)  
    })  

def test_init(sample_data):  
    afe = AutoFeatureEngineer(sample_data, 'target')  
    assert afe.data.equals(sample_data)  
    assert afe.target_column == 'target'  
    assert afe.feature_matrix is None  
    assert afe.feature_defs is None  
    assert afe.custom_features == {}  
    assert afe.entityset is None  

def test_create_entity_set(sample_data):
    afe = AutoFeatureEngineer(sample_data, 'target')
    es = afe.create_entity_set(index_column='id', time_index='feature3')
    assert "data" in es.dataframe_dict.keys()
    assert es["data"].ww.index == 'id'  # 使用 Woodwork 确认索引
    assert 'feature3' in es["data"].columns  # 确保时间索引列还在

def test_generate_features(sample_data):  
    afe = AutoFeatureEngineer(sample_data, 'target')  
    afe.create_entity_set(index_column='id')  
    feature_matrix, feature_defs = afe.generate_features(max_depth=2)  
    assert isinstance(feature_matrix, pd.DataFrame)  
    assert len(feature_defs) > 0  
    assert afe.feature_matrix is not None  
    assert afe.feature_defs is not None  

def test_create_custom_feature(sample_data):  
    afe = AutoFeatureEngineer(sample_data, 'target')  
    afe.create_entity_set(index_column='id')  
    afe.generate_features()  
    afe.create_custom_feature('custom_feature', lambda x: x['feature1'] * 2)  
    assert 'custom_feature' in afe.feature_matrix.columns  

def test_get_important_features(sample_data):  
    afe = AutoFeatureEngineer(sample_data, 'target')  
    afe.create_entity_set(index_column='id')  
    afe.generate_features()  
    important_features = afe.get_important_features(n=5, method='correlation')  
    assert 0 < len(important_features) <= 5  
    assert all(isinstance(feature, str) for feature in important_features)  

def test_get_feature_types(sample_data):  
    afe = AutoFeatureEngineer(sample_data, 'target')  
    afe.create_entity_set(index_column='id')  
    afe.generate_features()  
    feature_types = afe.get_feature_types()  
    assert isinstance(feature_types, dict)  
    assert all(feature_type in ['numeric', 'categorical', 'datetime', 'object'] for feature_type in feature_types.values())  

def test_get_feature_descriptions(sample_data):  
    afe = AutoFeatureEngineer(sample_data, 'target')  
    afe.create_entity_set(index_column='id')  
    afe.generate_features()  
    feature_descriptions = afe.get_feature_descriptions()  
    assert isinstance(feature_descriptions, list)  
    assert all(isinstance(desc, str) for desc in feature_descriptions)  

def test_get_feature_matrix(sample_data):  
    afe = AutoFeatureEngineer(sample_data, 'target')  
    afe.create_entity_set(index_column='id')  
    afe.generate_features()  
    feature_matrix = afe.get_feature_matrix()  
    assert isinstance(feature_matrix, pd.DataFrame)  

def test_remove_low_information_features(sample_data):  
    afe = AutoFeatureEngineer(sample_data, 'target')  
    afe.create_entity_set(index_column='id')  
    afe.generate_features()  
    initial_columns = afe.feature_matrix.columns  
    removed_columns = afe.remove_low_information_features(threshold=0.95)  
    assert set(removed_columns).issubset(set(initial_columns))  
    assert all(column not in afe.feature_matrix.columns for column in removed_columns)  

def test_remove_highly_correlated_features(sample_data):  
    afe = AutoFeatureEngineer(sample_data, 'target')  
    afe.create_entity_set(index_column='id')  
    afe.generate_features()  
    initial_columns = afe.feature_matrix.columns  
    removed_columns = afe.remove_highly_correlated_features(threshold=0.9)  
    assert set(removed_columns).issubset(set(initial_columns))  
    assert all(column not in afe.feature_matrix.columns for column in removed_columns)  

def test_normalize_features(sample_data):  
    afe = AutoFeatureEngineer(sample_data, 'target')  
    afe.create_entity_set(index_column='id')  
    afe.generate_features()  
    afe.normalize_features(method='standard')  
    numeric_features = afe.feature_matrix.select_dtypes(include=[np.number]).columns  
    assert all(afe.feature_matrix[numeric_features].mean().abs() < 1e-6)  
    assert all((afe.feature_matrix[numeric_features].std() - 1).abs() < 1e-6)  

def test_encode_categorical_features(sample_data):  
    afe = AutoFeatureEngineer(sample_data, 'target')  
    afe.create_entity_set(index_column='id')  
    afe.generate_features()  
    initial_categorical_columns = afe.feature_matrix.select_dtypes(include=['object']).columns  
    afe.encode_categorical_features(method='onehot')  
    assert all(column not in afe.feature_matrix.columns for column in initial_categorical_columns)  
    assert all(afe.feature_matrix.dtypes != 'object')  

def test_generate_features_without_entity_set():
    afe = AutoFeatureEngineer(pd.DataFrame({'A': [1, 2, 3]}), 'A')
    with pytest.raises(ValueError, match="Entity set not created"):
        afe.generate_features()

def test_generate_features_with_custom_primitives():
    print(f"Featuretools version: {ft.__version__}")
    np.random.seed(42)

    # 创建主数据表
    data = pd.DataFrame({
        'id': range(100),
        'A': np.random.rand(100),
        'B': np.random.rand(100),
        'target': np.random.randint(0, 2, 100)
    })

    # 创建子数据表
    child_data = pd.DataFrame({
        'child_id': range(200),
        'parent_id': np.random.randint(0, 100, size=200),
        'value': np.random.rand(200)
    })

    # 创建实体集并添加数据表
    es = ft.EntitySet(id='test')
    es = es.add_dataframe(dataframe_name='data', dataframe=data, index='id')
    es = es.add_dataframe(dataframe_name='child', dataframe=child_data, index='child_id')

    # 创建并添加关系
    es = es.add_relationship('data', 'id', 'child', 'parent_id')

    print(f"EntitySet:\n{es}")

    # 初始化 AutoFeatureEngineer
    afe = AutoFeatureEngineer(data, 'target')
    afe.entityset = es

    # 使用自定义原语生成特征
    custom_primitives = ['mean', 'max', 'min', 'std']
    feature_matrix, feature_defs = afe.generate_features(
        max_depth=2,
        primitives=custom_primitives
    )

    # 打印调试信息
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Number of feature definitions: {len(feature_defs)}")
    print(f"Feature names: {[f.get_name() for f in feature_defs]}")

    # 断言
    assert len(feature_defs) > len(data.columns), f"Expected more than {len(data.columns)} features, got {len(feature_defs)}"
    
    # 检查是否生成了包含自定义原语的特征
    for primitive in custom_primitives:
        assert any(primitive.upper() in f.get_name().upper() for f in feature_defs), f"No {primitive.upper()} features generated"

    # 检查特征矩阵的列数是否与特征定义的数量相匹配
    assert len(feature_matrix.columns) == len(feature_defs), "Feature matrix columns don't match feature definitions"
    
@pytest.mark.parametrize("method", ['correlation', 'mutual_info', 'mutual_info_regression', 'mutual_info_classif'])
def test_get_important_features_methods(method):
    data = pd.DataFrame({
        'id': range(100),
        'A': np.random.rand(100),
        'B': np.random.rand(100),
        'C': np.random.choice(['X', 'Y', 'Z'], 100),
        'target': np.random.randint(0, 2, 100)
    })
    afe = AutoFeatureEngineer(data, 'target')
    afe.create_entity_set('id')
    afe.generate_features()
    important_features = afe.get_important_features(method=method)
    assert len(important_features) > 0

def test_get_important_features_invalid_method():
    afe = AutoFeatureEngineer(pd.DataFrame({'A': [1, 2, 3], 'target': [0, 1, 0]}), 'target')
    afe.create_entity_set('A')
    afe.generate_features()
    with pytest.raises(ValueError, match="Invalid method"):
        afe.get_important_features(method='invalid_method')

def test_remove_low_information_features():
    data = pd.DataFrame({
        'id': range(100),
        'A': np.random.rand(100),
        'B': [1] * 100,  # Low information feature
        'target': np.random.randint(0, 2, 100)
    })
    afe = AutoFeatureEngineer(data, 'target')
    afe.create_entity_set('id')
    afe.generate_features()
    
    print(f"Feature matrix before removal:\n{afe.feature_matrix}")
    print(f"Unique value counts before removal: {afe.feature_matrix.nunique()}")
    
    removed = afe.remove_low_information_features(threshold=0.9)
    
    print(f"Feature matrix after removal:\n{afe.feature_matrix}")
    print(f"Unique value counts after removal: {afe.feature_matrix.nunique()}")
    print(f"Removed features: {removed}")
    
    assert 'B' in removed

def test_remove_highly_correlated_features():
    np.random.seed(42)
    A = np.random.rand(100)
    data = pd.DataFrame({
        'id': range(100),
        'A': A,
        'B': np.random.rand(100),
        'C': np.random.rand(100),
        'D': A + np.random.normal(0, 0.1, 100),  # 高度相关于 A
        'target': np.random.randint(0, 2, 100)
    })

    afe = AutoFeatureEngineer(data, 'target')
    afe.create_entity_set('id')
    afe.generate_features()

    print(f"Correlation matrix before removal:\n{afe.feature_matrix.corr()}")
    
    removed = afe.remove_highly_correlated_features(threshold=0.9)
    
    print(f"Correlation matrix after removal:\n{afe.feature_matrix.corr()}")
    print(f"Removed features: {removed}")
    
    assert len(removed) > 0
    assert 'D' in removed  # 'D' 应该被移除，因为它与 'A' 高度相关

def test_normalize_features_invalid_method():
    afe = AutoFeatureEngineer(pd.DataFrame({'A': [1, 2, 3], 'target': [0, 1, 0]}), 'target')
    afe.create_entity_set('A')
    afe.generate_features()
    with pytest.raises(ValueError, match="Invalid normalization method"):
        afe.normalize_features(method='invalid')

def test_encode_categorical_features_invalid_method():
    afe = AutoFeatureEngineer(pd.DataFrame({'A': ['X', 'Y', 'Z'], 'target': [0, 1, 0]}), 'target')
    afe.create_entity_set('A')
    afe.generate_features()
    with pytest.raises(ValueError, match="Invalid encoding method"):
        afe.encode_categorical_features(method='invalid')                        

if __name__ == "__main__":  
    pytest.main()