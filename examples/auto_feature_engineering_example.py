# examples/auto_feature_engineering_example.py

from src.core.auto_feature_engineering import AutoFeatureEngineer
import pandas as pd
from sklearn.datasets import load_boston

def main():
    # 加载波士顿房价数据集作为示例
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    
    # 初始化自动特征工程器
    auto_fe = AutoFeatureEngineer(df)
    
    print("Starting automatic feature engineering process...")
    
    # 生成新特征
    feature_matrix, feature_defs = auto_fe.generate_features()
    
    print(f"Original number of features: {df.shape[1]}")
    print(f"Number of features after auto engineering: {feature_matrix.shape[1]}")
    
    # 显示一些新生成的特征
    print("\nSample of new features:")
    for i, feature_def in enumerate(feature_defs[:5]):
        print(f"{i+1}. {feature_def}")

if __name__ == "__main__":
    main()