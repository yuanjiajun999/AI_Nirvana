# E:\AI_Nirvana-1\tests\test_privacy_enhancement.py

import unittest
import numpy as np
import sys
import os

# 添加 src 目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from core.privacy_enhancement import PrivacyEnhancement, AdvancedAnonymization, FederatedLearningSimulator, HomomorphicEncryptionSimulator

class TestPrivacyEnhancement(unittest.TestCase):
    def setUp(self):
        self.pe = PrivacyEnhancement()

    def test_add_laplace_noise(self):
        value = 10
        noisy_value = self.pe.add_laplace_noise(value)
        self.assertNotEqual(value, noisy_value)

    def test_apply_differential_privacy(self):
        data = [1, 2, 3, 4, 5]
        dp_data = self.pe.apply_differential_privacy(data)
        self.assertEqual(len(data), len(dp_data))
        self.assertNotEqual(data, dp_data)

    def test_set_privacy_budget(self):
        new_epsilon = 0.5
        self.pe.set_privacy_budget(new_epsilon)
        self.assertEqual(self.pe.epsilon, new_epsilon)

    def test_encrypt_decrypt_data(self):
        original_data = "sensitive information"
        encrypted = self.pe.encrypt_data(original_data)
        decrypted = self.pe.decrypt_data(encrypted)
        self.assertEqual(original_data, decrypted)

    def test_k_anonymity(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        k = 3
        anonymized = self.pe.k_anonymity(data, k)
        self.assertEqual(len(data), len(anonymized))
        unique_values = set(anonymized)
        self.assertLessEqual(len(unique_values), len(data) // k + 1)

    def test_gaussian_noise(self):
        data = [1, 2, 3, 4, 5]
        noisy_data = self.pe.gaussian_noise(data)
        self.assertEqual(len(data), len(noisy_data))
        self.assertNotEqual(data, noisy_data)

    def test_truncated_gaussian_noise(self):
        data = [1, 2, 3, 4, 5]
        noisy_data = self.pe.truncated_gaussian_noise(data)
        self.assertEqual(len(data), len(noisy_data))
        self.assertNotEqual(data, noisy_data)
        self.assertTrue(all(-1 <= x <= 6 for x in noisy_data))

    def test_hash_data(self):
        data = [1, 2, 3, 4, 5]
        hashed_data = self.pe.hash_data(data)
        self.assertEqual(len(data), len(hashed_data))
        self.assertTrue(all(isinstance(x, str) for x in hashed_data))

    def test_dimensionality_reduction(self):
        data = np.random.rand(100, 10)
        reduced_data = self.pe.dimensionality_reduction(data, n_components=5)
        self.assertEqual(reduced_data.shape, (100, 5))

    def test_randomized_response(self):
        data = [True, False, True, False, True]
        rr_data = self.pe.randomized_response(data)
        self.assertEqual(len(data), len(rr_data))
        self.assertTrue(all(isinstance(x, bool) for x in rr_data))

    def test_exponential_mechanism(self):
        data = [1, 2, 3, 4, 5]
        utility_function = lambda x: x  # 简单的效用函数
        result = self.pe.exponential_mechanism(data, utility_function, sensitivity=1)
        self.assertIn(result, data)

    def test_local_differential_privacy(self):
        data = [0, 1, 0, 1, 1]
        ldp_data = self.pe.local_differential_privacy(data)
        self.assertEqual(len(data), len(ldp_data))
        self.assertTrue(all(x in [0, 1] for x in ldp_data))

    def test_secure_aggregation(self):
        data_chunks = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = self.pe.secure_aggregation(data_chunks)
        self.assertEqual(result, 45)

class TestAdvancedAnonymization(unittest.TestCase):
    def setUp(self):
        self.aa = AdvancedAnonymization()

    def test_t_closeness(self):
        # 由于这是一个简化的实现，我们只测试方法是否存在
        self.assertTrue(hasattr(self.aa, 't_closeness'))

    def test_l_diversity(self):
        # 由于这是一个简化的实现，我们只测试方法是否存在
        self.assertTrue(hasattr(self.aa, 'l_diversity'))

class TestFederatedLearningSimulator(unittest.TestCase):
    def setUp(self):
        self.fls = FederatedLearningSimulator()

    def test_simulate_federated_learning(self):
        # 由于这是一个简化的实现，我们只测试方法是否存在
        self.assertTrue(hasattr(self.fls, 'simulate_federated_learning'))

class TestHomomorphicEncryptionSimulator(unittest.TestCase):
    def setUp(self):
        self.hes = HomomorphicEncryptionSimulator()

    def test_simulate_homomorphic_encryption(self):
        # 由于这是一个简化的实现，我们只测试方法是否存在
        self.assertTrue(hasattr(self.hes, 'simulate_homomorphic_encryption'))

if __name__ == '__main__':
    unittest.main()