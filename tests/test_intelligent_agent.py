# tests/test_intelligent_agent.py

import os
import unittest
import numpy as np
from src.core.intelligent_agent import IntelligentAgent, DQNAgent

class MockEnvironment:
    def __init__(self):
        self.state = np.zeros(4)
        self.step_count = 0

    def reset(self):
        self.state = np.random.rand(4)
        self.step_count = 0
        return self.state

    def step(self, action):
        self.state = np.random.rand(4)
        self.step_count += 1
        reward = np.random.rand()
        done = self.step_count >= 100
        return self.state, reward, done, {}

class TestIntelligentAgent(unittest.TestCase):
    def setUp(self):
        self.agent = DQNAgent(state_size=4, action_size=2)
        self.environment = MockEnvironment()

    def test_initialization(self):
        self.assertEqual(self.agent.state_size, 4)
        self.assertEqual(self.agent.action_size, 2)
        self.assertEqual(len(self.agent.memory), 0)

    def test_perceive(self):
        observation = np.array([1, 2, 3, 4])
        perceived_state = self.agent.perceive(observation)
        np.testing.assert_array_equal(perceived_state, np.reshape(observation, [1, self.agent.state_size]))

    def test_decide(self):
        state = np.array([[1, 2, 3, 4]])
        action = self.agent.decide(state)
        self.assertIn(action, [0, 1])

    def test_act(self):
        state = np.array([[1, 2, 3, 4]])
        action = self.agent.act(state)
        self.assertIsInstance(action, (int, np.integer))  # 允许 numpy 整数类型
        self.assertTrue(0 <= action < self.agent.action_size)
    
    def test_remember(self):
        state = np.array([1, 2, 3, 4])
        action = 1
        reward = 0.5
        next_state = np.array([2, 3, 4, 5])
        done = False
        self.agent.remember(state, action, reward, next_state, done)
        self.assertEqual(len(self.agent.memory), 1)

    def test_replay(self):
        # Fill memory with some experiences
        for _ in range(50):
            state = np.random.rand(4)
            action = np.random.randint(2)
            reward = np.random.rand()
            next_state = np.random.rand(4)
            done = np.random.choice([True, False])
            self.agent.remember(state, action, reward, next_state, done)

        initial_epsilon = self.agent.epsilon
        self.agent.replay(32)
        self.assertLess(self.agent.epsilon, initial_epsilon)

    def test_save_and_load(self):
        weights_file = "test_weights"  # 移除 .h5 扩展名
        self.agent.save(weights_file)
        new_agent = DQNAgent(state_size=4, action_size=2)
        new_agent.load(f"{weights_file}.weights.h5")
    
        # Check if weights are the same
        for w1, w2 in zip(self.agent.model.get_weights(), new_agent.model.get_weights()):
            np.testing.assert_array_almost_equal(w1, w2)
    
        # Clean up
        os.remove(f"{weights_file}.weights.h5")

    def test_run(self):
        class MockEnvironment:
            def __init__(self):
                self.state = np.zeros(4)
                self.step_count = 0

            def reset(self):
                self.state = np.random.rand(4)
                self.step_count = 0
                return self.state

            def step(self, action):
                self.state = np.random.rand(4)
                self.step_count += 1
                reward = np.random.rand()
                done = self.step_count >= 10
                return self.state, reward, done, {}

        env = MockEnvironment()
        episodes = 2
        max_steps = 5
        self.agent.run(env, episodes, max_steps)
        self.assertTrue(len(self.agent.memory) > 0)

    def test_perceive_different_shapes(self):
        observations = [
            np.array([1, 2, 3, 4]),
            np.array([[1, 2, 3, 4]]),
            np.array([[[1, 2, 3, 4]]])
        ]
        for obs in observations:
            perceived = self.agent.perceive(obs)
            assert perceived.shape == (1, self.agent.state_size)

    def test_decide_exploration_exploitation(self):
        state = np.array([[1, 2, 3, 4]])
        self.agent.epsilon = 1.0  # Always explore
        action = self.agent.decide(state)
        assert 0 <= action < self.agent.action_size
        
        self.agent.epsilon = 0.0  # Always exploit
        action = self.agent.decide(state)
        assert 0 <= action < self.agent.action_size

    def test_act(self):
        state = np.array([[1, 2, 3, 4]])
        action = self.agent.act(state)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < self.agent.action_size

    def test_run_early_termination(self):
        class EarlyTerminationEnv:
            def reset(self):
                return np.zeros(4)
            def step(self, action):
                return np.zeros(4), 0, True, {}

        env = EarlyTerminationEnv()
        self.agent.run(env, episodes=1, max_steps=100)
        # 这至少覆盖了提前终止的分支

    def test_run_max_steps(self):
        class MaxStepsEnv:
            def reset(self):
                return np.zeros(4)
            def step(self, action):
                return np.zeros(4), 0, False, {}

        env = MaxStepsEnv()
        self.agent.run(env, episodes=1, max_steps=10)
        # 这覆盖了达到 max_steps 的情况

    def test_replay_different_batch_sizes(self):
        # 填充记忆
        for _ in range(100):
            state = np.random.rand(4)
            action = np.random.randint(2)
            reward = np.random.rand()
            next_state = np.random.rand(4)
            done = np.random.choice([True, False])
            self.agent.remember(state, action, reward, next_state, done)

        # 测试不同的批次大小
        for batch_size in [32, 64, 128]:
            self.agent.replay(batch_size)
        
        # 测试批次大小大于记忆大小的情况
        self.agent.replay(200)

    def test_abstract_methods_implementation(self):
        assert hasattr(self.agent, 'perceive')
        assert hasattr(self.agent, 'decide')
        assert hasattr(self.agent, 'act')
        assert hasattr(self.agent, 'remember')
 
        # 测试这些方法是否可调用
        state = np.array([1, 2, 3, 4])
        assert callable(self.agent.perceive)
        perceived_state = self.agent.perceive(state)
        assert perceived_state.shape == (1, self.agent.state_size)

        assert callable(self.agent.decide)
        action = self.agent.decide(perceived_state)
        assert isinstance(action, (int, np.integer))

        assert callable(self.agent.act)
        act_result = self.agent.act(perceived_state)
        assert isinstance(act_result, (int, np.integer))

        assert callable(self.agent.remember)
        self.agent.remember(state, action, 0, state, False)
        assert len(self.agent.memory) > 0


if __name__ == '__main__':
    unittest.main()