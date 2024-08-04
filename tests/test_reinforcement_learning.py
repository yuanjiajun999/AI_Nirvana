import unittest
import numpy as np
import gym
import tensorflow as tf
import os
from unittest.mock import Mock, patch
import time
import pytest
from src.core.reinforcement_learning import (
    BaseAgent, DQNAgent, A2CAgent, PPOAgent, SACAgent, TD3Agent, DDPGAgent,
    MultiAgentRL, HierarchicalRL, CuriosityDrivenRL, MetaLearningAgent,
    create_environment, train_agent, evaluate_agent,
    plot_learning_curve, save_agent, load_agent, parallel_train_agents,
    process_state
)

class TestReinforcementLearning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\nStarting Reinforcement Learning tests. This may take a while...")
        cls.start_time = time.time()

    @classmethod
    def tearDownClass(cls):
        elapsed_time = time.time() - cls.start_time
        print(f"\nAll tests completed in {elapsed_time:.2f} seconds.")
    
    def setUp(self):
        self.env = gym.make('CartPole-v1')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.continuous_env = gym.make('Pendulum-v1')
        self.continuous_state_size = self.continuous_env.observation_space.shape[0]
        self.continuous_action_size = self.continuous_env.action_space.shape[0]

    def _process_state(self, state):
        return np.array(state).reshape(1, -1)
    
    def test_dqn_agent(self):
        agent = DQNAgent(self.state_size, self.action_size)
        self._test_discrete_agent(agent)

    def test_a2c_agent(self):
        agent = A2CAgent(self.state_size, self.action_size)
        self._test_discrete_agent(agent)

    def test_ppo_agent(self):
        agent = PPOAgent(self.state_size, self.action_size)
        self._test_discrete_agent(agent)

    def test_sac_agent(self):
        agent = SACAgent(self.continuous_state_size, self.continuous_action_size)
        self._test_continuous_agent(agent)

    def test_td3_agent(self):
        agent = TD3Agent(self.continuous_state_size, self.continuous_action_size)
        self._test_continuous_agent(agent)

    def test_ddpg_agent(self):
        agent = DDPGAgent(self.continuous_state_size, self.continuous_action_size)
        self._test_continuous_agent(agent)

    def test_multi_agent_rl(self):
        env = gym.make('CartPole-v1')
        multi_agent = MultiAgentRL(num_agents=2, state_size=env.observation_space.shape[0], action_size=env.action_space.n)
    
        state, _ = env.reset()
        states = [self._process_state(state) for _ in range(2)]
    
        for _ in range(10):  # Reduce training steps to 10 for faster testing
            actions = multi_agent.act(states)
            next_state, reward, done, _, _ = env.step(actions[0])  # Use only the first action
            next_states = [self._process_state(next_state) for _ in range(2)]
        
            losses = multi_agent.train(states, actions, [reward, reward], next_states, [done, done])
        
            self.assertTrue(all(loss is not None for loss in losses))
        
            if done:
                state, _ = env.reset()
                states = [self._process_state(state) for _ in range(2)]
            else:
                states = next_states
    
        env.close()

    def test_hierarchical_rl(self):
        agent = HierarchicalRL(self.state_size, self.action_size)
        state = self._process_state(self.env.reset()[0])
        action, option = agent.act(state)
        self.assertIsInstance(action, (int, np.integer))
        self.assertIsInstance(option, (int, np.integer))

        next_state, reward, done, truncated, _ = self.env.step(action)
        next_state = self._process_state(next_state)
        done = done or truncated
        loss = agent.train(state, action, reward, next_state, done, option)
        if loss is not None:
            self._assert_loss(loss)

    def test_curiosity_driven_rl(self):
        agent = CuriosityDrivenRL(self.state_size, self.action_size)
        self._test_discrete_agent(agent)

    def test_meta_learning_agent(self):
        agent = MetaLearningAgent(self.state_size, self.action_size)
        episodes = [
            [(self._process_state(self.env.reset()[0]), 0, 1.0, self._process_state(self.env.step(0)[0]), False)] * 10
            for _ in range(5)
        ]
        agent.meta_train(episodes)

        state = self._process_state(self.env.reset()[0])
        adapted_q_values = agent.adapt(episodes[0])
        action = agent.act(state, adapted_q_values)
        self.assertIsInstance(action, (int, np.integer))

    def test_create_environment(self):
        env = create_environment('CartPole-v1')
        self.assertIsInstance(env, gym.Env)

    def test_train_agent(self):
        agent = DQNAgent(self.state_size, self.action_size)
        trained_agent = train_agent(agent, self.env, num_episodes=5, max_steps_per_episode=200)
        self.assertIsInstance(trained_agent, DQNAgent)

    def test_evaluate_agent(self):
        agent = DQNAgent(self.state_size, self.action_size)
        avg_reward, std_reward = evaluate_agent(agent, self.env, num_episodes=5)
        self.assertIsInstance(avg_reward, float)
        self.assertIsInstance(std_reward, float)

    def _test_discrete_agent(self, agent):
        state = self._process_state(self.env.reset()[0])
        for _ in range(5):  # Run a few steps to fill the memory
            action = agent.act(state)
            next_state, reward, done, truncated, _ = self.env.step(action)
            next_state = self._process_state(next_state)
            loss = agent.train(state, action, reward, next_state, done)
            if loss is not None:
                self._assert_loss(loss)
            state = next_state
            if done:
                break
        self.assertIsInstance(action, (int, np.integer))
        self.assertTrue(0 <= action < self.action_size)

    def _test_continuous_agent(self, agent):
        state = self._process_state(self.continuous_env.reset()[0])
        for _ in range(5):  # Run a few steps to fill the memory
            action = agent.act(state)
            next_state, reward, done, truncated, _ = self.continuous_env.step(action)
            next_state = self._process_state(next_state)
            loss = agent.train(state, action, reward, next_state, done)
            if loss is not None:
                self._assert_loss(loss)
            state = next_state
            if done:
                break
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape, (self.continuous_action_size,))

    def test_plot_learning_curve(self):
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        plot_learning_curve(rewards, window_size=2)
        # This test just ensures the function runs without error

    def test_save_and_load_agent(self):
        agent = DQNAgent(self.state_size, self.action_size)
        filename = 'test_agent.pkl'
        save_agent(agent, filename)
        loaded_agent = load_agent(filename)
        self.assertIsInstance(loaded_agent, DQNAgent)
        os.remove(filename)  # Clean up

    def test_parallel_train_agents(self):
        agents = [DQNAgent(self.state_size, self.action_size) for _ in range(2)]
        envs = [gym.make('CartPole-v1') for _ in range(2)]
        trained_agents = parallel_train_agents(agents, envs, num_episodes=5, max_steps_per_episode=200)
        self.assertEqual(len(trained_agents), 2)
        for agent in trained_agents:
            self.assertIsInstance(agent, DQNAgent)

    def test_dqn_agent_memory(self):
        agent = DQNAgent(self.state_size, self.action_size)
        state = self.env.reset()[0]
        action = agent.act(state)
        next_state, reward, done, truncated, _ = self.env.step(action)
        done = done or truncated
        agent.remember(state, action, reward, next_state, done)
        self.assertEqual(len(agent.memory), 1)

    def test_dqn_agent_replay(self):
        agent = DQNAgent(self.state_size, self.action_size, batch_size=1)
        state = self.env.reset()[0]
        action = agent.act(state)
        next_state, reward, done, truncated, _ = self.env.step(action)
        done = done or truncated
        agent.remember(state, action, reward, next_state, done)
        loss = agent.replay(1)
        self._assert_loss(loss)

    def test_a2c_agent_build_model(self):
        agent = A2CAgent(self.state_size, self.action_size)
        self.assertIsInstance(agent.actor, tf.keras.Model)
        self.assertIsInstance(agent.critic, tf.keras.Model)

    def test_ppo_agent_build_model(self):
        agent = PPOAgent(self.state_size, self.action_size)
        self.assertIsInstance(agent.actor, tf.keras.Model)
        self.assertIsInstance(agent.critic, tf.keras.Model)

    def test_sac_agent_build_model(self):
        agent = SACAgent(self.continuous_state_size, self.continuous_action_size)
        self.assertIsInstance(agent.actor, tf.keras.Model)
        self.assertIsInstance(agent.critic1, tf.keras.Model)
        self.assertIsInstance(agent.critic2, tf.keras.Model)

    def test_td3_agent_build_model(self):
        agent = TD3Agent(self.continuous_state_size, self.continuous_action_size)
        self.assertIsInstance(agent.actor, tf.keras.Model)
        self.assertIsInstance(agent.critic1, tf.keras.Model)
        self.assertIsInstance(agent.critic2, tf.keras.Model)

    def test_ddpg_agent_build_model(self):
        agent = DDPGAgent(self.continuous_state_size, self.continuous_action_size)
        self.assertIsInstance(agent.actor, tf.keras.Model)
        self.assertIsInstance(agent.critic, tf.keras.Model)

    def test_curiosity_driven_rl_build_model(self):
        agent = CuriosityDrivenRL(self.state_size, self.action_size)
        self.assertIsInstance(agent.forward_model, tf.keras.Model)

    def test_meta_learning_agent_build_model(self):
        agent = MetaLearningAgent(self.state_size, self.action_size)
        self.assertIsInstance(agent.meta_model, tf.keras.Model)

    def test_multi_agent_rl_create_agents(self):
        multi_agent = MultiAgentRL(2, self.state_size, self.action_size, agent_type='DQN')
        self.assertEqual(len(multi_agent.agents), 2)
        for agent in multi_agent.agents:
            self.assertIsInstance(agent, DQNAgent)

    def test_hierarchical_rl_create_agents(self):
        agent = HierarchicalRL(self.state_size, self.action_size, num_options=4)
        self.assertIsInstance(agent.meta_controller, DQNAgent)
        self.assertEqual(len(agent.options), 4)
        for option in agent.options:
            self.assertIsInstance(option, DQNAgent)

    def _assert_loss(self, loss):
        self.assertIsInstance(loss, (float, np.floating))
        self.assertGreater(loss, 0)
        self.assertLess(loss, 1e6)

    def test_process_state(self):
        agent = DQNAgent(4, 2)

        # Test with numpy array
        state_array = np.array([1, 2, 3, 4])
        processed_state = agent.process_state(state_array)
        self.assertIsInstance(processed_state, np.ndarray)
        self.assertEqual(processed_state.shape, (1, 4))

        # Test with list
        state_list = [1, 2, 3, 4]
        processed_state = agent.process_state(state_list)
        self.assertIsInstance(processed_state, np.ndarray)
        self.assertEqual(processed_state.shape, (1, 4))

        # Test with tuple
        state_tuple = (1, 2, 3, 4)
        processed_state = agent.process_state(state_tuple)
        self.assertIsInstance(processed_state, np.ndarray)
        self.assertEqual(processed_state.shape, (1, 4))

        # Test with dictionary
        state_dict = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        processed_state = agent.process_state(state_dict)
        self.assertIsInstance(processed_state, np.ndarray)
        self.assertEqual(processed_state.shape, (1, 4))

        # Test with single value
        state_single = 1
        processed_state = agent.process_state(state_single)
        self.assertIsInstance(processed_state, np.ndarray)
        self.assertEqual(processed_state.shape, (1, 1))

    def test_dqn_train(self):
        env = gym.make('CartPole-v1')
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    
        state, _ = env.reset()
        state = self._process_state(state)
    
        for _ in range(10):  # Reduce training steps to 10 for faster testing
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = self._process_state(next_state)
        
            loss = agent.train(state, action, reward, next_state, done)
        
            self.assertIsNotNone(loss)
        
            if done:
                state, _ = env.reset()
                state = self._process_state(state)
            else:
                state = next_state
    
        env.close()
        
    def test_a2c_train(self):
        env = gym.make('CartPole-v1')
        agent = A2CAgent(env.observation_space.shape[0], env.action_space.n)
    
        state, _ = env.reset()
        state = agent.process_state(state)
    
        for _ in range(100):  # Train for 100 steps
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = agent.process_state(next_state)
        
            loss = agent.train(state, action, reward, next_state, done)
        
            self.assertIsNotNone(loss)
        
            if done:
                state, _ = env.reset()
                state = agent.process_state(state)
            else:
                state = next_state
    
        env.close()

    def _test_agent(self, agent_class, name):
        print(f"\nTesting {name}...")
        start_time = time.time()

        if agent_class in [SACAgent, TD3Agent, DDPGAgent]:
            env = self.continuous_env
            state_size = self.continuous_state_size
            action_size = self.continuous_action_size
        else:
            env = self.env
            state_size = self.state_size
            action_size = self.action_size

        agent = agent_class(state_size, action_size)
        
        state, _ = env.reset()
        state = self._process_state(state)
        
        for step in range(100):  # Keep 100 steps for thorough testing
            action = agent.act(state)
            if isinstance(action, np.ndarray):
                action = action.flatten()
            next_state, reward, done, _, _ = env.step(action)
            next_state = self._process_state(next_state)
            
            loss = agent.train(state, action, reward, next_state, done)
            
            self.assertIsNotNone(loss)
            
            if done:
                state, _ = env.reset()
                state = self._process_state(state)
            else:
                state = next_state

            if (step + 1) % 20 == 0:
                print(f"  Step {step + 1}/100 completed")
        
        env.close()
        elapsed_time = time.time() - start_time
        print(f"{name} test completed in {elapsed_time:.2f} seconds")

    def test_dqn_agent(self):
        self._test_agent(DQNAgent, "DQN Agent")

    def test_a2c_agent(self):
        self._test_agent(A2CAgent, "A2C Agent")

    def test_ppo_agent(self):
        self._test_agent(PPOAgent, "PPO Agent")

    def test_sac_agent(self):
        self._test_agent(SACAgent, "SAC Agent")

    def test_td3_agent(self):
        self._test_agent(TD3Agent, "TD3 Agent")

    def test_ddpg_agent(self):
        self._test_agent(DDPGAgent, "DDPG Agent")

    def test_multi_agent_rl(self):
        env = gym.make('CartPole-v1')
        multi_agent = MultiAgentRL(num_agents=2, state_size=env.observation_space.shape[0], action_size=env.action_space.n)
    
        state, _ = env.reset()
        states = [self._process_state(state) for _ in range(2)]
    
        for _ in range(10):  # Reduce training steps to 10 for faster testing
            actions = multi_agent.act(states)
            next_state, reward, done, _, _ = env.step(actions[0])  # Use only the first action
            next_states = [self._process_state(next_state) for _ in range(2)]
        
            losses = multi_agent.train(states, actions, [reward, reward], next_states, [done, done])
        
            self.assertTrue(all(loss is not None for loss in losses))
        
            if done:
                state, _ = env.reset()
                states = [self._process_state(state) for _ in range(2)]
            else:
                states = next_states
    
        env.close()

    def test_hierarchical_rl(self):
        env = gym.make('CartPole-v1')
        agent = HierarchicalRL(env.observation_space.shape[0], env.action_space.n)
    
        state, _ = env.reset()
        state = agent.process_state(state)
    
        for _ in range(100):  # Train for 100 steps
            action, option = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = agent.process_state(next_state)
        
            agent.train(state, action, reward, next_state, done, option)
        
            if done:
                state, _ = env.reset()
                state = agent.process_state(state)
            else:
                state = next_state
    
        env.close()

    def test_curiosity_driven_rl(self):
        env = gym.make('CartPole-v1')
        agent = CuriosityDrivenRL(env.observation_space.shape[0], env.action_space.n)
    
        state, _ = env.reset()
        state = agent.process_state(state)
    
        for _ in range(100):  # Train for 100 steps
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = agent.process_state(next_state)
        
            agent.train(state, action, reward, next_state, done)
        
            if done:
                state, _ = env.reset()
                state = agent.process_state(state)
            else:
                state = next_state
    
        env.close()

    def test_meta_learning_agent(self):
        env = gym.make('CartPole-v1')
        agent = MetaLearningAgent(env.observation_space.shape[0], env.action_space.n)
    
        episodes = []
        for _ in range(5):  # Generate 5 episodes
            episode = []
            state, _ = env.reset()
            state = agent.process_state(state)
            for _ in range(100):  # Max 100 steps per episode
                action = np.random.randint(env.action_space.n)
                next_state, reward, done, _, _ = env.step(action)
                next_state = agent.process_state(next_state)
                episode.append((state, action, reward, next_state, done))
                if done:
                    break
                state = next_state
            episodes.append(episode)
    
        agent.meta_train(episodes)
    
        # Test adaptation
        adapted_q_values = agent.adapt(episodes[0])
        self.assertEqual(adapted_q_values.shape, (env.action_space.n,))
    
        # Test act with adapted q-values
        state, _ = env.reset()
        state = agent.process_state(state)
        action = agent.act(state, adapted_q_values)
        self.assertTrue(0 <= action < env.action_space.n)
    
        env.close()

    def test_helper_functions(self):
        env = create_environment('CartPole-v1')
        self.assertIsInstance(env, gym.Env)
    
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
        trained_agent = train_agent(agent, env, num_episodes=10, max_steps_per_episode=100)
        self.assertIsInstance(trained_agent, DQNAgent)
    
        avg_reward, std_reward = evaluate_agent(trained_agent, env, num_episodes=10)
        self.assertIsInstance(avg_reward, float)
        self.assertIsInstance(std_reward, float)
    
        save_agent(trained_agent, 'test_agent.pkl')
        loaded_agent = load_agent('test_agent.pkl')
        self.assertIsInstance(loaded_agent, DQNAgent)
    
        # Clean up
        os.remove('test_agent.pkl')
    
        env.close()

    def test_parallel_training(self):
        num_agents = 2
        envs = [create_environment('CartPole-v1') for _ in range(num_agents)]
        agents = [DQNAgent(envs[0].observation_space.shape[0], envs[0].action_space.n) for _ in range(num_agents)]
    
        trained_agents = parallel_train_agents(agents, envs, num_episodes=10, max_steps_per_episode=100)
        self.assertEqual(len(trained_agents), num_agents)
        self.assertTrue(all(isinstance(agent, DQNAgent) for agent in trained_agents))
    
        for env in envs:
            env.close()

    def test_process_state(self):
        # Test with different input types
        self.assertEqual(process_state(1).shape, (1, 1))
        self.assertEqual(process_state([1, 2, 3]).shape, (1, 3))
        self.assertEqual(process_state(np.array([1, 2, 3])).shape, (1, 3))
        self.assertEqual(process_state({'a': 1, 'b': 2}).shape, (1, 2))

    def test_dqn_edge_cases(self):
        agent = DQNAgent(4, 2, batch_size=32)
        state = np.array([0, 0, 0, 0])
        next_state = np.array([1, 0, 0, 0])
        
        # Test training with insufficient memory
        loss = agent.train(state, 0, 1, next_state, False)
        self.assertEqual(loss, 0)  # Should return 0 when memory is insufficient

        # Fill the memory
        for _ in range(32):
            agent.remember(state, 0, 1, next_state, False)

        # Now train should return a non-zero loss
        loss = agent.train(state, 0, 1, next_state, False)
        self.assertGreater(loss, 0)

    def test_sac_agent(self):
        agent = SACAgent(4, 2)
        state = np.array([0, 0, 0, 0])
    
        # Test sample_action
        action, log_pi = agent.sample_action(state)
        self.assertEqual(action.shape, (1, 2))
        self.assertEqual(log_pi.shape, (1, 1))

        # Test act
        action = agent.act(state)
        self.assertEqual(action.shape, (2,))

        # Test update_target_networks
        old_weights = agent.target_critic1.get_weights()
        agent.update_target_networks()
        new_weights = agent.target_critic1.get_weights()
        self.assertFalse(np.array_equal(old_weights[0], new_weights[0]))  # Weights should be updated

    def test_multi_agent_rl(self):
        # Test creation of different agent types
        agent_types = ['DQN', 'A2C', 'PPO', 'SAC', 'TD3', 'DDPG']
        for agent_type in agent_types:
            multi_agent = MultiAgentRL(2, 4, 2, agent_type=agent_type)
            self.assertEqual(len(multi_agent.agents), 2)
            self.assertIsInstance(multi_agent.agents[0], eval(f"{agent_type}Agent"))

        # Test unsupported agent type
        with self.assertRaises(ValueError):
            MultiAgentRL(2, 4, 2, agent_type='UNSUPPORTED')

    def test_helper_functions(self):
        # Test create_environment
        env = create_environment('CartPole-v1')
        self.assertIsInstance(env, gym.Env)

        # Test plot_learning_curve
        rewards = [1, 2, 3, 4, 5]
        plot_learning_curve(rewards)  # This should not raise any error

        # Test save_agent and load_agent
        agent = DQNAgent(4, 2)
        save_agent(agent, 'test_agent.pkl')
        loaded_agent = load_agent('test_agent.pkl')
        self.assertIsInstance(loaded_agent, DQNAgent)
        os.remove('test_agent.pkl')  # Clean up

        # Test parallel_train_agents
        agents = [DQNAgent(4, 2) for _ in range(2)]
        envs = [create_environment('CartPole-v1') for _ in range(2)]
        results = parallel_train_agents(agents, envs, num_episodes=2, max_steps_per_episode=10)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(agent, DQNAgent) for agent in results))
        
    def test_base_agent_methods(self):
        class ConcreteAgent(BaseAgent):
            def act(self, state):
                return 0
            def train(self, state, action, reward, next_state, done):
                return 0
        agent = ConcreteAgent()
        self.assertEqual(agent.act(np.array([0, 0, 0, 0])), 0)
        self.assertEqual(agent.train(np.array([0, 0, 0, 0]), 0, 1, np.array([1, 1, 1, 1]), False), 0)

    def test_dqn_edge_cases(self):
        agent = DQNAgent(4, 2, memory_size=10, batch_size=5)
        state = np.array([0, 0, 0, 0])
        for _ in range(15):  # 填充并溢出记忆
            agent.remember(state, 0, 0, state, False)
        loss = agent.train(state, 0, 0, state, False)
        self.assertIsNotNone(loss)

    def test_sac_specific_methods(self):
        agent = SACAgent(4, 2)
        state = np.array([0, 0, 0, 0])
        action, log_pi = agent.sample_action(state)
        self.assertEqual(action.shape, (1, 2))
        self.assertEqual(log_pi.shape, (1, 1))

    def test_multi_agent_rl_methods(self):
        multi_agent = MultiAgentRL(2, 4, 2)
        states = [np.array([0, 0, 0, 0]) for _ in range(2)]
        actions = multi_agent.act(states)
        self.assertEqual(len(actions), 2)
        losses = multi_agent.train(states, actions, [0, 0], states, [False, False])
        self.assertEqual(len(losses), 2)

    def test_sac_agent_detailed(self):
        agent = SACAgent(4, 2)
        state = np.array([0, 0, 0, 0])
        next_state = np.array([1, 1, 1, 1])
    
        # 测试 sample_action 方法
        action, log_pi = agent.sample_action(state)
        self.assertEqual(action.shape, (1, 2))
        self.assertEqual(log_pi.shape, (1, 1))
    
        # 测试 act 方法
        action = agent.act(state)
        self.assertEqual(action.shape, (2,))
    
        # 测试 train 方法
        loss = agent.train(state, action, 1.0, next_state, False)
        self.assertIsNotNone(loss)
    
        # 测试 update_target_networks 方法
        agent.update_target_networks()
    
        # 如果可能，测试内部方法
        if hasattr(agent, '_compute_intrinsic_reward'):
            intrinsic_reward = agent._compute_intrinsic_reward(state, action, next_state)
            self.assertIsInstance(intrinsic_reward, float)
    
    def test_dqn_memory_full(self):
        agent = DQNAgent(4, 2, memory_size=10)
        state = np.array([0, 0, 0, 0])
        for _ in range(15):  # 填充并溢出记忆
            agent.remember(state, 0, 0, state, False)
        self.assertEqual(len(agent.memory), 10)
        loss = agent.train(state, 0, 0, state, False)
        self.assertIsNotNone(loss)

    def test_multi_agent_rl_edge_cases(self):
        multi_agent = MultiAgentRL(2, 4, 2)
        states = [np.array([0, 0, 0, 0]) for _ in range(2)]
        actions = multi_agent.act(states)
        self.assertEqual(len(actions), 2)
        losses = multi_agent.train(states, actions, [0, 0], states, [True, False])
        self.assertEqual(len(losses), 2)

    def test_create_environment(self):
        env = create_environment('CartPole-v1')
        self.assertIsNotNone(env)
        self.assertEqual(env.observation_space.shape[0], 4)
        self.assertEqual(env.action_space.n, 2)

    def test_plot_learning_curve(self):
        rewards = [1, 2, 3, 4, 5]
        plot_learning_curve(rewards, window_size=2)
        # 这个测试主要是确保函数运行时不会抛出异常

    def test_save_and_load_agent(self):
        agent = DQNAgent(4, 2)
        filename = 'test_agent.pkl'
        save_agent(agent, filename)
        loaded_agent = load_agent(filename)
        self.assertIsInstance(loaded_agent, DQNAgent)
        os.remove(filename)  # 清理测试文件

    @patch('gym.make')
    def test_agent_with_mocked_environment(self, mock_make):
        mock_env = Mock()
        mock_env.observation_space.shape = (4,)
        mock_env.action_space.n = 2
        mock_make.return_value = mock_env
        
        agent = DQNAgent(4, 2)
        state = np.array([0, 0, 0, 0])
        mock_env.reset.return_value = (state, {})
        mock_env.step.return_value = (state, 1.0, False, False, {})
        
        # 确保在训练循环开始前调用 reset
        initial_state, _ = mock_env.reset()
        
        for _ in range(10):
            action = agent.act(initial_state)
            next_state, reward, done, _, _ = mock_env.step(action)
            agent.train(initial_state, action, reward, next_state, done)
            if done:
                initial_state, _ = mock_env.reset()  # 如果回合结束，重置环境
            else:
                initial_state = next_state
        
        self.assertTrue(mock_env.step.called)
        self.assertTrue(mock_env.reset.called)
        self.assertGreaterEqual(mock_env.reset.call_count, 1)  # 确保 reset 至少被调用一次
        self.assertGreaterEqual(mock_env.step.call_count, 10)  # 确保 step 被调用至少 10 次

    def test_base_agent(self):
        class ConcreteAgent(BaseAgent):
            def act(self, state):
                return 0
            def train(self, state, action, reward, next_state, done):
                return 0

        agent = ConcreteAgent()
        state = np.array([0, 0, 0, 0])
        self.assertEqual(agent.act(state), 0)
        self.assertEqual(agent.train(state, 0, 1, state, False), 0)
        processed_state = agent.process_state(state)
        self.assertEqual(processed_state.shape, (1, 4))

    def test_dqn_edge_cases(self):
        agent = DQNAgent(4, 2, memory_size=5, batch_size=2)
        state = np.array([0, 0, 0, 0])

        # 测试记忆溢出
        for _ in range(10):
            agent.remember(state, 0, 1, state, False)
        self.assertEqual(len(agent.memory), 5)

        # 测试不同的 epsilon 值
        agent.epsilon = 0
        self.assertEqual(agent.act(state), np.argmax(agent.model.predict(state.reshape(1, -1))[0]))
        agent.epsilon = 1
        self.assertIn(agent.act(state), [0, 1])

        # 测试 replay 方法
        loss = agent.replay(2)
        self.assertIsNotNone(loss)

        # 测试目标模型更新
        old_weights = agent.target_model.get_weights()
        agent.update_target_model()
        new_weights = agent.target_model.get_weights()
        self.assertTrue(np.allclose(old_weights[0], new_weights[0], atol=1e-6))  # 使用 np.allclose 而不是 np.array_equal

        # Test training with insufficient memory
        initial_loss = agent.train(state, 0, 1, state, False)
        self.assertIsNotNone(initial_loss)  # Loss should be a number, not None

        # Fill memory and train multiple times
        losses = []
        for _ in range(10):
            agent.remember(state, 0, 1, state, False)
            loss = agent.train(state, 0, 1, state, False)
            losses.append(loss)

        # Check if loss changes over time
        self.assertNotEqual(min(losses), max(losses), "Loss should change during training")

        # Test epsilon decay
        initial_epsilon = agent.epsilon
        for _ in range(100):
            state = np.random.rand(4)  # 生成随机状态
            action = agent.act(state)
            next_state = np.random.rand(4)  # 生成随机下一状态
            reward = np.random.rand()  # 生成随机奖励
            done = bool(np.random.randint(2))  # 随机生成是否结束
            agent.train(state, action, reward, next_state, done)

        self.assertLess(agent.epsilon, initial_epsilon, "Epsilon should decay")

        # Test target model update
        old_weights = agent.target_model.get_weights()
        for _ in range(agent.update_target_frequency):
            agent.train(state, 0, 1, state, False)
        new_weights = agent.target_model.get_weights()
        self.assertFalse(np.allclose(old_weights[0], new_weights[0], atol=1e-6), "Target model should update")

        print(f"Initial loss: {initial_loss}")
        print(f"Min loss: {min(losses)}, Max loss: {max(losses)}")
        print(f"Initial epsilon: {initial_epsilon}, Final epsilon: {agent.epsilon}")
        print(f"Train count: {agent.get_train_count()}")

    def test_sac_agent_detailed(self):
        agent = SACAgent(4, 2)
        state = np.array([0, 0, 0, 0])
        next_state = np.array([1, 1, 1, 1])
    
        # Test sample_action
        action, log_pi = agent.sample_action(state)
        self.assertEqual(action.shape, (1, 2))
        self.assertEqual(log_pi.shape, (1, 1))
    
        # Test act
        action = agent.act(state)
        self.assertEqual(action.shape, (2,))
    
        # Test train
        loss = agent.train(state, action, 1.0, next_state, False)
        self.assertIsNotNone(loss)
    
        # Test update_target_networks
        old_weights = agent.target_critic1.get_weights()[0]
        agent.update_target_networks()
        new_weights = agent.target_critic1.get_weights()[0]
        self.assertFalse(np.array_equal(old_weights, new_weights))

    def test_helper_functions(self):  
        # Test create_environment  
        env = create_environment('CartPole-v1')  
        self.assertIsNotNone(env)  
        self.assertIsInstance(env, gym.Env)  

        # Test train_agent  
        agent = DQNAgent(4, 2)  
        trained_agent = train_agent(agent, env, num_episodes=2, max_steps_per_episode=10)  
        self.assertIsInstance(trained_agent, DQNAgent)  

        # Test evaluate_agent  
        avg_reward, std_reward = evaluate_agent(trained_agent, env, num_episodes=2)  
        self.assertIsInstance(avg_reward, float)  
        self.assertIsInstance(std_reward, float)  

        # Test plot_learning_curve  
        rewards = [1, 2, 3, 4, 5]  
        plot_learning_curve(rewards, window_size=2)  

        # Test save_agent and load_agent  
        filename = 'test_agent.pkl'  
        save_agent(trained_agent, filename)  
        loaded_agent = load_agent(filename)  
        self.assertIsInstance(loaded_agent, DQNAgent)  
        os.remove(filename)  # Clean up  

        # Test parallel_train_agents  
        agents = [DQNAgent(4, 2) for _ in range(2)]  
        envs = [create_environment('CartPole-v1') for _ in range(2)]  
        results = parallel_train_agents(agents, envs, num_episodes=2, max_steps_per_episode=10)  
        self.assertEqual(len(results), 2)

    def test_advanced_agents(self):
        for AgentClass in [SACAgent, TD3Agent, DDPGAgent]:
            agent = AgentClass(4, 2)
            state = np.array([0, 0, 0, 0])
            next_state = np.array([1, 1, 1, 1])
        
            # Test act method
            action = agent.act(state)
            self.assertEqual(action.shape, (2,))
        
            # Test train method
            loss = agent.train(state, action, 1.0, next_state, False)
            self.assertIsNotNone(loss)
        
            # Test update_target_networks method
            if hasattr(agent, 'update_target_networks'):
                if isinstance(agent, SACAgent):
                    old_weights = agent.target_critic1.get_weights()
                else:
                    old_weights = agent.actor_target.get_weights()
                agent.update_target_networks()
                if isinstance(agent, SACAgent):
                    new_weights = agent.target_critic1.get_weights()
                else:
                    new_weights = agent.actor_target.get_weights()
                self.assertFalse(np.array_equal(old_weights[0], new_weights[0]))
    
    def test_complex_agents(self):
        # Test MultiAgentRL
        multi_agent = MultiAgentRL(2, 4, 2)
        states = [np.array([0, 0, 0, 0]) for _ in range(2)]
        actions = multi_agent.act(states)
        self.assertEqual(len(actions), 2)
        losses = multi_agent.train(states, actions, [0, 0], states, [False, False])
        self.assertEqual(len(losses), 2)

        # Test HierarchicalRL
        hier_agent = HierarchicalRL(4, 2)
        state = np.array([0, 0, 0, 0])
        action, option = hier_agent.act(state)
        self.assertIsInstance(action, (int, np.integer))
        self.assertIsInstance(option, (int, np.integer))
        hier_agent.train(state, action, 0, state, False, option)

        # Test CuriosityDrivenRL
        curiosity_agent = CuriosityDrivenRL(4, 2)
        action = curiosity_agent.act(state)
        self.assertIsInstance(action, (int, np.integer))
        curiosity_agent.train(state, action, 0, state, False)

    def test_a2c_edge_cases(self):
        agent = A2CAgent(4, 2)
        state = np.array([0, 0, 0, 0])
        action = 0
        reward = 1.0
        next_state = np.array([1, 1, 1, 1])
        done = False

        # 测试训练过程
        loss = agent.train(state, action, reward, next_state, done)
        self.assertIsInstance(loss, float)

    def test_ppo_edge_cases(self):
        agent = PPOAgent(4, 2)
        states = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
        actions = np.array([0, 1])
        rewards = np.array([1.0, 0.5])
        next_states = np.array([[1, 1, 1, 1], [0, 0, 0, 0]])
        dones = np.array([False, True])

        # 测试训练过程
        loss = agent.train(states, actions, rewards, next_states, dones)
        self.assertIsInstance(loss, (float, np.float32, np.float64))  # 允许 numpy 的浮点类型

    def test_sac_edge_cases(self):
        agent = SACAgent(4, 2)
        state = np.array([0, 0, 0, 0])
        action = np.array([0.5, -0.5])
        reward = 1.0
        next_state = np.array([1, 1, 1, 1])
        done = False

        # 测试 sample_action 方法
        sampled_action, log_pi = agent.sample_action(state)
        self.assertEqual(sampled_action.shape, (1, 2))
        self.assertEqual(log_pi.shape, (1, 1))

        # 测试训练过程
        loss = agent.train(state, action, reward, next_state, done)
        self.assertIsInstance(loss, float)

        # 测试目标网络更新
        old_weights = agent.target_critic1.get_weights()
        agent.update_target_networks()
        new_weights = agent.target_critic1.get_weights()
        self.assertFalse(np.array_equal(old_weights[0], new_weights[0]))

    def test_td3_edge_cases(self):
        agent = TD3Agent(4, 2)
        state = np.array([0, 0, 0, 0])
        action = np.array([0.5, -0.5])
        reward = 1.0
        next_state = np.array([1, 1, 1, 1])
        done = False

        # 测试训练过程
        loss = agent.train(state, action, reward, next_state, done)
        self.assertIsInstance(loss, float)

        # 测试目标网络更新
        old_weights = agent.actor_target.get_weights()
        agent.update_target_networks()
        new_weights = agent.actor_target.get_weights()
        self.assertFalse(np.array_equal(old_weights[0], new_weights[0]))

        # 测试策略更新频率
        agent.total_it = agent.policy_freq - 1
        loss_with_policy_update = agent.train(state, action, reward, next_state, done)
        self.assertIsInstance(loss_with_policy_update, float)

    def test_multi_agent_rl_edge_cases(self):
        multi_agent = MultiAgentRL(2, 4, 2, agent_type='DQN')
        states = [np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1])]
        actions = multi_agent.act(states)
        self.assertEqual(len(actions), 2)
        losses = multi_agent.train(states, actions, [1.0, 0.5], states, [False, False])
        self.assertEqual(len(losses), 2)

    def test_hierarchical_rl_edge_cases(self):
        agent = HierarchicalRL(4, 2)
        state = np.array([0, 0, 0, 0])
        action, option = agent.act(state)
        self.assertIsInstance(action, (int, np.integer))
        self.assertIsInstance(option, (int, np.integer))
        agent.train(state, action, 1.0, state, False, option)

if __name__ == '__main__':
    unittest.main()