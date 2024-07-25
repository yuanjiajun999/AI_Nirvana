import unittest

import numpy as np
import pandas as pd

from src.core.auto_feature_engineering import AutoFeatureEngineer


class TestAutoFeatureEngineer(unittest.TestCase):
    def setUp(self):
        # 创建一个简单的数据集用于测试
        np.random.seed(42)
        self.data = pd.DataFrame(
            {
                "id": range(100),
                "A": np.random.rand(100),
                "B": np.random.randint(0, 5, 100),
                "timestamp": pd.date_range(start="1/1/2021", periods=100),
            }
        )
        self.auto_fe = AutoFeatureEngineer(self.data)

    def test_create_entity_set(self):
        self.auto_fe.create_entity_set()
        self.assertIsNotNone(self.auto_fe.entity_set)

        self.assertEqual(
            len(self.auto_fe.entity_set.entities), 1
        )  # 只有一个实体（数据框）

    def test_generate_features(self):
        self.auto_fe.create_entity_set()
        feature_matrix, feature_defs = self.auto_fe.generate_features()
        self.assertIsInstance(feature_matrix, pd.DataFrame)
        self.assertTrue(len(feature_matrix) == len(self.data))
        self.assertTrue(len(feature_defs) > 0)

    def test_get_important_features(self):
        self.auto_fe.create_entity_set()
        self.auto_fe.generate_features()
        important_features = self.auto_fe.get_important_features(n=5)
        self.assertIsInstance(important_features, list)
        self.assertTrue(0 < len(important_features) <= 5)


if __name__ == "__main__":
    unittest.main()
