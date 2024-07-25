import unittest
import numpy as np
from src.core.reinforcement_learning import ReinforcementLearningAgent


class TestReinforcementLearningAgent(unittest.TestCase):
    def setUp(self):
        self.agent = ReinforcementLearningAgent(state_size=4, action_size=2)

    def test_act(self):
        state = np.random.rand(4)
        action = self.agent.act(state)
        self.assertIn(action, [0, 1])

    def test_train(self):
        state = np.random.rand(4)
        action = 1
        reward = 1.0
        next_state = np.random.rand(4)
        done = False

        # This test just ensures the method runs without error
        try:
            self.agent.train(state, action, reward, next_state, done)
        except Exception as e:
            self.fail(f"train() raised {type(e).__name__} unexpectedly!")


if __name__ == "__main__":
    unittest.main()
