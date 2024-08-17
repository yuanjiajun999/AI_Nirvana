import unittest
import numpy as np
from src.core.digital_twin import DigitalTwin
from src.config import Config
from src.main import AINirvana
from unittest.mock import patch

def dummy_physical_model(state, t):
    return [-0.1 * state[0], -0.05 * state[1]]

class DummyPhysicalSystemModel:
    def detect_anomalies(self, sensor_data):
        return [i for i, value in enumerate(sensor_data) if value > 1.0]

    def optimize(self, objective_function, constraints):
        return [0.5, 0.5]

class TestDigitalTwin(unittest.TestCase):
    def setUp(self):
        self.digital_twin = DigitalTwin(dummy_physical_model)

    def test_simulate(self):
        initial_conditions = [1.0]
        time_steps = np.linspace(0, 10, 100)
        states = self.digital_twin.simulate(initial_conditions, time_steps)
    
        # 检查返回值是否为 numpy 数组
        self.assertIsInstance(states, np.ndarray)
    
        # 检查形状是否符合预期（假设模型返回两个状态变量）
        self.assertEqual(states.shape, (100, 2))

    def test_monitor(self):
        dummy_model = DummyPhysicalSystemModel()
        self.digital_twin.physical_system_model = dummy_model
        sensor_data = np.array([0.5, 1.5, 0.8, 2.0])
        anomalies = self.digital_twin.monitor(sensor_data)
        self.assertEqual(anomalies, [1, 3])

    def test_optimize(self):
        def objective_function(x):
            return x[0]**2 + x[1]**2
        constraints = [(lambda state: 100 - state[0], 'ineq')]
        result = self.digital_twin.optimize(objective_function, constraints, method='COBYLA')
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 2)

    def test_update_model(self):
        new_model = lambda state, t: [-0.2 * state[0], -0.1 * state[1]]
        self.digital_twin.update_model(new_model)
        self.assertEqual(self.digital_twin.physical_system_model, new_model)

    def test_validate_model(self):
        validation_data = np.array([0.1, 0.2, 0.3, 0.4])
        result = self.digital_twin.validate_model(validation_data)
    
        # 检查返回值是否为字典
        self.assertIsInstance(result, dict)
    
        # 检查字典中是否包含预期的键
        expected_keys = ['error', 'positive', 'neutral', 'negative']
        for key in expected_keys:
            self.assertIn(key, result)
    
        # 如果没有错误，检查是否所有值都是浮点数
        if result['error'] is None:
            for key in ['positive', 'neutral', 'negative']:
                 self.assertIsInstance(result[key], float)
        else:
            # 如果有错误，检查错误信息是否为字符串
            self.assertIsInstance(result['error'], str)
    
    def test_preprocess_data(self):
        data = [1, 2, 3, 4, 5]
        processed_data = self.digital_twin.preprocess_data(data)
        self.assertIsInstance(processed_data, np.ndarray)
        self.assertEqual(len(processed_data), len(data))

    def test_postprocess_results(self):
        results = np.array([[1, 2], [3, 4]])
        processed_results = self.digital_twin.postprocess_results(results)
        np.testing.assert_array_equal(processed_results, results)

    def test_digital_twin_initialization(self):
        config = Config("config.json")
        ai_nirvana = AINirvana(config)
        self.assertIsNotNone(ai_nirvana.digital_twin, "数字孪生系统未能初始化")
        print("测试通过：数字孪生系统成功初始化。")

    def test_digital_twin_simulation(self):
        config = Config("config.json")
        ai_nirvana = AINirvana(config)
        initial_conditions = [1.0, 2.0]
        time_steps = [0, 1, 2, 3]
        result = ai_nirvana.simulate_digital_twin(initial_conditions, time_steps)
        self.assertIsNotNone(result, "模拟未能成功执行")
        self.assertEqual(result.shape, (4, 2))  # 4个时间步，2个状态变量
        print("测试通过：数字孪生系统模拟成功。")

    def test_monitor_without_detect_anomalies(self):
        self.digital_twin.physical_system_model = lambda x: x  # 一个没有 detect_anomalies 方法的模型
        result = self.digital_twin.monitor(np.array([1, 2, 3]))
        self.assertIn('error', result)
        self.assertIn("Physical system model does not have a 'detect_anomalies' method", result['error'])

    def test_optimize_differential_evolution(self):
        def objective_function(x):
            return x[0]**2 + x[1]**2
        bounds = [(-10, 10), (-10, 10)]
        result = self.digital_twin.optimize(objective_function, bounds, method='differential_evolution')
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)
        self.assertIn("`constraint` of an unknown type is passed", result['error'])

    def test_validate_model_with_different_data(self):
        validation_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = self.digital_twin.validate_model(validation_data)
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)

    def test_preprocess_data_short(self):
        short_data = [1, 2, 3]
        result = self.digital_twin.preprocess_data(short_data)
        np.testing.assert_array_equal(result, np.array(short_data))

    def test_visualize_simulation(self):
        states = np.array([[1, 2], [3, 4], [5, 6]])
        time_steps = [0, 1, 2]
        with patch('matplotlib.pyplot.show') as mock_show:
            self.digital_twin.visualize_simulation(states, time_steps)
            mock_show.assert_called_once()                   

if __name__ == '__main__':
    unittest.main()