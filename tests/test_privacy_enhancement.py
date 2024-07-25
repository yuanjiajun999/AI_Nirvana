import unittest
from src.core.privacy_enhancement import PrivacyEnhancement


class TestPrivacyEnhancement(unittest.TestCase):
    def setUp(self):
        self.privacy_enhancer = PrivacyEnhancement(epsilon=1.0)

    def test_add_laplace_noise(self):
        value = 10.0
        noisy_value = self.privacy_enhancer.add_laplace_noise(value)
        self.assertNotEqual(value, noisy_value)

    def test_apply_differential_privacy(self):
        data = [1, 2, 3, 4, 5]
        private_data = self.privacy_enhancer.apply_differential_privacy(data)
        self.assertEqual(len(data), len(private_data))
        self.assertNotEqual(data, private_data)

    def test_set_privacy_budget(self):
        new_epsilon = 0.5
        self.privacy_enhancer.set_privacy_budget(new_epsilon)
        self.assertEqual(self.privacy_enhancer.epsilon, new_epsilon)


if __name__ == "__main__":
    unittest.main()
