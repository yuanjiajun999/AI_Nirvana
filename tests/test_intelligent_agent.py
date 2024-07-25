import unittest
from src.core.intelligent_agent import IntelligentAgent


class TestIntelligentAgent(unittest.TestCase):
    def setUp(self):
        self.agent = IntelligentAgent()

    def test_perceive(self):
        observation = "The sky is blue"
        result = self.agent.perceive(observation)
        self.assertIsNotNone(result)

    def test_decide(self):
        state = "current state"
        action = self.agent.decide(state)
        self.assertIsNotNone(action)

    def test_act(self):
        action = "walk forward"
        result = self.agent.act(action)
        self.assertIsNotNone(result)

    def test_run(self):
        result = self.agent.run()
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
