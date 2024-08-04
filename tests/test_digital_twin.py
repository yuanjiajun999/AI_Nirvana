import unittest
import numpy as np
from src.core.digital_twin import DigitalTwin

def dummy_physical_model(state, t):
    return -0.1 * state

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
        self.assertEqual(states.shape, (100, 1))

    def test_monitor(self):
        dummy_model = DummyPhysicalSystemModel()
        self.digital_twin.physical_system_model = dummy_model
        sensor_data = np.array([0.5, 1.5, 0.8, 2.0])
        anomalies = self.digital_twin.monitor(sensor_data)
        self.assertEqual(anomalies, [1, 3])

    def test_optimize(self):
        dummy_model = DummyPhysicalSystemModel()
        self.digital_twin.physical_system_model = dummy_model
        def objective_function(x):
            return x[0]**2 + x[1]**2
        constraints = [(-1, 1), (-1, 1)]
        optimal_params = self.digital_twin.optimize(objective_function, constraints)
        self.assertEqual(optimal_params, [0.5, 0.5])

    def test_update_model(self):
        new_model = lambda state, t: -0.2 * state
        self.digital_twin.update_model(new_model)
        self.assertEqual(self.digital_twin.physical_system_model, new_model)

    def test_validate_model(self):
        validation_data = np.array([0.1, 0.2, 0.3, 0.4])
        accuracy = self.digital_twin.validate_model(validation_data)
        self.assertIsInstance(accuracy, float)

if __name__ == '__main__':
    unittest.main()