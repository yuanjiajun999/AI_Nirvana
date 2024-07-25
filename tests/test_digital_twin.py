import unittest
import numpy as np
from src.core.digital_twin import DigitalTwin


class TestDigitalTwin(unittest.TestCase):
    def setUp(self):
        def simple_model(state, t):
            return np.array([state[1], -state[0]])  # Simple harmonic oscillator

        self.digital_twin = DigitalTwin(simple_model)

    def test_simulate(self):
        initial_conditions = [1.0, 0.0]
        time_steps = np.linspace(0, 10, 100)
        states = self.digital_twin.simulate(initial_conditions, time_steps)
        self.assertEqual(states.shape, (100, 2))
        self.assertAlmostEqual(
            states[-1, 0], -1.0, delta=0.1
        )  # Approximate final position

    def test_monitor(self):
        sensor_data = np.random.rand(100, 2)
        anomalies = self.digital_twin.monitor(sensor_data)
        self.assertIsInstance(anomalies, list)

    def test_optimize(self):
        def objective_function(params):
            return np.sum(params**2)

        constraints = [(0, 1), (0, 1)]
        optimal_params = self.digital_twin.optimize(objective_function, constraints)
        self.assertEqual(len(optimal_params), 2)
        self.assertTrue(all(0 <= p <= 1 for p in optimal_params))


if __name__ == "__main__":
    unittest.main()
