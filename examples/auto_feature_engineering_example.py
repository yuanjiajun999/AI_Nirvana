import pandas as pd
import numpy as np
from src.core.auto_feature_engineering import AutoFeatureEngineer


def main():
    # 创建一个示例数据集
    data = pd.DataFrame(
        {
            "id": range(1000),
            "A": np.random.rand(1000),
            "B": np.random.randint(0, 5, 1000),
            "timestamp": pd.date_range(start="1/1/2021", periods=1000),
        }
    )

    # 初始化自动特征工程器
    auto_fe = AutoFeatureEngineer(data)

    # 创建实体集
    auto_fe.create_entity_set()

    # 生成特征
    feature_matrix, feature_defs = auto_fe.generate_features()

    print("Original data shape:", data.shape)
    print("Feature matrix shape:", feature_matrix.shape)

    # 获取重要特征
    important_features = auto_fe.get_important_features(n=10)
    print("\nTop 10 important features:")
    for feature in important_features:
        print(feature)


if __name__ == "__main__":
    main()
