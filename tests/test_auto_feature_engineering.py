import warnings  
import pytest  
import pandas as pd  
import numpy as np  
import featuretools as ft  
import sys  
import os  

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
    assert es["data"].index.name == 'id'
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

if __name__ == "__main__":  
    pytest.main()