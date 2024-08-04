import unittest
import numpy as np
from sklearn.datasets import make_classification, make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from src.core.active_learning import ActiveLearner
import matplotlib.pyplot as plt

class TestActiveLearner(unittest.TestCase):
    def setUp(self):
        # Create a mock classification dataset
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=2,
                                   n_redundant=10, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        self.active_learner = ActiveLearner(self.X_train, self.y_train, self.X_test, self.y_test, random_state=42)
        self.active_learner.model = RandomForestClassifier(random_state=42)

        # Create a small dataset for edge case testing
        self.X_small, self.y_small = make_classification(n_samples=10, n_features=5, random_state=42)

        # Create a multilabel dataset
        self.X_multilabel, self.y_multilabel = make_multilabel_classification(n_samples=100, n_features=10, n_classes=3, random_state=42)

    def test_initialization(self):
        self.assertEqual(self.active_learner.X_pool.shape, self.X_train.shape)
        self.assertEqual(self.active_learner.y_pool.shape, self.y_train.shape)
        self.assertEqual(len(self.active_learner.labeled_indices), 0)
        self.assertIsInstance(self.active_learner.model, RandomForestClassifier)

    def test_train(self):
        initial_samples = 100
        X_initial = self.X_train[:initial_samples]
        y_initial = self.y_train[:initial_samples]
        self.active_learner.train(X_initial, y_initial)
        self.assertTrue(self.active_learner.is_fitted)

    def test_predict(self):
        self.active_learner.train(self.X_train[:100], self.y_train[:100])
        predictions = self.active_learner.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))

    def test_uncertainty_sampling(self):
        n_samples = 10
        self.active_learner.train(self.X_train[:100], self.y_train[:100])
        selected_indices = self.active_learner.uncertainty_sampling(n_samples)
        self.assertEqual(len(selected_indices), n_samples)
        self.assertTrue(np.all(selected_indices < len(self.X_train)))

    def test_diversity_sampling(self):
        n_samples = 10
        selected_indices = self.active_learner.diversity_sampling(n_samples)
        self.assertEqual(len(selected_indices), n_samples)
        self.assertTrue(np.all(selected_indices < len(self.X_train)))

    def test_expected_model_change_sampling(self):
        n_samples = 5
        self.active_learner.train(self.X_train, self.y_train)
        selected_indices = self.active_learner.expected_model_change_sampling(n_samples)
        self.assertEqual(len(selected_indices), n_samples)
        self.assertTrue(all(isinstance(i, (int, np.integer)) for i in selected_indices))

    def test_density_weighted_sampling(self):
        n_samples = 10
        self.active_learner.train(self.X_train[:100], self.y_train[:100])
        selected_indices = self.active_learner.density_weighted_sampling(n_samples)
        self.assertEqual(len(selected_indices), n_samples)
        self.assertTrue(np.all(selected_indices < len(self.X_train)))

    def test_evaluate(self):
        self.active_learner.train(self.X_train[:100], self.y_train[:100])
        accuracy = self.active_learner.evaluate()
        self.assertGreater(accuracy, 0)
        self.assertLess(accuracy, 1)

    def test_active_learning_loop(self):
        initial_samples = 100
        n_iterations = 5
        samples_per_iteration = 20
        final_accuracy, accuracy_history = self.active_learner.active_learning_loop(
            initial_samples, n_iterations, samples_per_iteration)
        
        self.assertGreater(final_accuracy, 0)
        self.assertEqual(len(accuracy_history), n_iterations)
        self.assertTrue(all(0 <= acc <= 1 for acc in accuracy_history))

    def test_set_and_get_model(self):
        new_model = SVC(probability=True, random_state=42)
        self.active_learner.set_model(new_model)
        retrieved_model = self.active_learner.get_model()
        self.assertIsInstance(retrieved_model, SVC)
        self.assertFalse(self.active_learner.is_fitted)

    def test_different_evaluation_metrics(self):
        self.active_learner.train(self.X_train[:100], self.y_train[:100])
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        for metric in metrics:
            score = self.active_learner.evaluate(metric=metric)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

        with self.assertRaises(ValueError):
            self.active_learner.evaluate(metric='invalid_metric')

    def test_get_pool_and_labeled_size(self):
        initial_pool_size = self.active_learner.get_pool_size()
        initial_labeled_size = self.active_learner.get_labeled_size()
        
        self.active_learner.active_learning_loop(initial_samples=100, n_iterations=3, samples_per_iteration=20)
        
        final_pool_size = self.active_learner.get_pool_size()
        final_labeled_size = self.active_learner.get_labeled_size()
        
        self.assertLess(final_pool_size, initial_pool_size)
        self.assertGreater(final_labeled_size, initial_labeled_size)

    def test_committee_creation_and_sampling(self):
        n_samples = 10
        initial_samples = 100

        # 首先标记一些数据
        self.active_learner.label_samples(np.arange(initial_samples))
    
        # 训练模型
        X_labeled, y_labeled = self.active_learner.get_labeled_data()
        self.active_learner.train(X_labeled, y_labeled)

        # 创建委员会
        self.active_learner.create_committee(n_models=3)

        # 进行委员会不确定性采样
        selected_indices = self.active_learner.committee_uncertainty_sampling(n_samples)

        self.assertEqual(len(selected_indices), n_samples)
        self.assertTrue(np.all(selected_indices < len(self.X_train)))

    def test_batch_mode(self):
        initial_samples = 100
        n_iterations = 5
        samples_per_iteration = 20
        final_accuracy, accuracy_history = self.active_learner.active_learning_loop(
            initial_samples, n_iterations, samples_per_iteration, batch_mode=True)
        
        self.assertGreater(final_accuracy, 0)
        self.assertEqual(len(accuracy_history), n_iterations)

    def test_active_learning_loop_with_different_strategies(self):
        strategies = ['uncertainty', 'diversity', 'expected_model_change', 'density_weighted']
        for strategy in strategies:
            final_accuracy, accuracy_history = self.active_learner.active_learning_loop(
                initial_samples=50, n_iterations=3, samples_per_iteration=10, strategy=strategy
            )
            self.assertIsInstance(final_accuracy, float)
            self.assertEqual(len(accuracy_history), 3)

    def test_invalid_strategy(self):
        with self.assertRaises(ValueError):
            self.active_learner.active_learning_loop(
                initial_samples=50, n_iterations=3, samples_per_iteration=10, strategy='invalid_strategy'
            )

    def test_update_pool(self):
        initial_pool_size = self.active_learner.get_pool_size()
        self.active_learner.update_pool(np.array([0, 1, 2]))
        self.assertEqual(self.active_learner.get_pool_size(), initial_pool_size - 3)

    def test_get_labeled_data(self):
        self.active_learner.label_samples(np.array([0, 1, 2]))
        X_labeled, y_labeled = self.active_learner.get_labeled_data()
        self.assertEqual(len(X_labeled), 3)
        self.assertEqual(len(y_labeled), 3)

    def test_predict_without_training(self):
        with self.assertRaises(ValueError):
            self.active_learner.predict(self.X_test)

    def test_small_dataset(self):
        small_active_learner = ActiveLearner(self.X_small, self.y_small, self.X_small, self.y_small)
        small_active_learner.model = RandomForestClassifier(random_state=42)
        final_accuracy, accuracy_history = small_active_learner.active_learning_loop(
            initial_samples=5, n_iterations=2, samples_per_iteration=2)
        self.assertIsInstance(final_accuracy, float)
        self.assertEqual(len(accuracy_history), 2)

    def test_multilabel_handling(self):
        multilabel_learner = ActiveLearner(self.X_multilabel, self.y_multilabel, self.X_multilabel, self.y_multilabel)
        multilabel_learner.model = RandomForestClassifier(random_state=42)
        handled_y = multilabel_learner.handle_multilabel(self.y_multilabel)
        self.assertEqual(handled_y.shape, self.y_multilabel.shape)

    def test_plot_learning_curve(self):
        accuracy_history = [0.5, 0.6, 0.7, 0.8]
        self.active_learner.plot_learning_curve(accuracy_history)
        # This test just ensures the method runs without error
        # You might want to save the plot and check if the file exists

    def test_empty_pool(self):
        empty_learner = ActiveLearner(np.array([]), np.array([]), self.X_test, self.y_test)
        with self.assertRaises(ValueError):
            empty_learner.uncertainty_sampling(10)

    def test_single_class_data(self):
        X_single_class = np.random.rand(100, 10)
        y_single_class = np.zeros(100)
        single_class_learner = ActiveLearner(X_single_class, y_single_class, X_single_class, y_single_class)
        single_class_learner.model = RandomForestClassifier(random_state=42)
        single_class_learner.train(X_single_class, y_single_class)
        accuracy = single_class_learner.evaluate()
        self.assertEqual(accuracy, 1.0)

if __name__ == '__main__':
    unittest.main()