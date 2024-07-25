import unittest

import numpy as np

from src.core.active_learning import ActiveLearner


class TestActiveLearner(unittest.TestCase):
    def setUp(self):
        # 创建一个简单的数据集用于测试
        np.random.seed(42)
        self.X = np.random.rand(100, 5)
        self.y = np.random.randint(0, 2, 100)
        self.X_pool, self.y_pool = self.X[:80], self.y[:80]
        self.X_test, self.y_test = self.X[80:], self.y[80:]

        self.active_learner = ActiveLearner(
            self.X_pool, self.y_pool, self.X_test, self.y_test
        )

    def test_uncertainty_sampling(self):
        n_samples = 5
        selected_indices = self.active_learner.uncertainty_sampling(n_samples)
        self.assertEqual(len(selected_indices), n_samples)
        self.assertTrue(all(0 <= idx < len(self.X_pool) for idx in selected_indices))

    def test_train(self):
        X_train = self.X_pool[:10]
        y_train = self.y_pool[:10]
        self.active_learner.train(X_train, y_train)
        # 检查模型是否已经被训练（即，模型的属性是否已被设置）
        self.assertIsNotNone(getattr(self.active_learner.model, "classes_", None))

    def test_evaluate(self):
        X_train = self.X_pool[:10]
        y_train = self.y_pool[:10]
        self.active_learner.train(X_train, y_train)
        accuracy = self.active_learner.evaluate()
        self.assertIsInstance(accuracy, float)
        self.assertTrue(0 <= accuracy <= 1)

    def test_active_learning_loop(self):
        initial_samples = 10
        n_iterations = 3
        samples_per_iteration = 5
        final_accuracy = self.active_learner.active_learning_loop(
            initial_samples, n_iterations, samples_per_iteration
        )
        self.assertIsInstance(final_accuracy, float)
        self.assertTrue(0 <= final_accuracy <= 1)


if __name__ == "__main__":
    unittest.main()
