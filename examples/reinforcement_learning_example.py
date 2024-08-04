# reinforcement_learning_examples.py

import gym
import numpy as np
from src.core.reinforcement_learning import (
    DQNAgent, A2CAgent, PPOAgent, SACAgent, TD3Agent, DDPGAgent,
    MultiAgentRL, HierarchicalRL, CuriosityDrivenRL, MetaLearningAgent,
    create_environment, train_agent, evaluate_agent,
    plot_learning_curve, save_agent, load_agent, parallel_train_agents
)

def dqn_example():
    print("DQN Agent Example")
    env = create_environment('CartPole-v1')
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    trained_agent = train_agent(agent, env, num_episodes=200, max_steps_per_episode=500)
    avg_reward, _ = evaluate_agent(trained_agent, env, num_episodes=100)
    print(f"DQN Average Reward: {avg_reward}")

def a2c_example():
    print("A2C Agent Example")
    env = create_environment('LunarLander-v2')
    agent = A2CAgent(env.observation_space.shape[0], env.action_space.n)
    trained_agent = train_agent(agent, env, num_episodes=300, max_steps_per_episode=1000)
    avg_reward, _ = evaluate_agent(trained_agent, env, num_episodes=100)
    print(f"A2C Average Reward: {avg_reward}")

def ppo_example():
    print("PPO Agent Example")
    env = create_environment('BipedalWalker-v3')
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0])
    trained_agent = train_agent(agent, env, num_episodes=500, max_steps_per_episode=1600)
    avg_reward, _ = evaluate_agent(trained_agent, env, num_episodes=100)
    print(f"PPO Average Reward: {avg_reward}")

def sac_example():
    print("SAC Agent Example")
    env = create_environment('Pendulum-v1')
    agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0])
    trained_agent = train_agent(agent, env, num_episodes=300, max_steps_per_episode=200)
    avg_reward, _ = evaluate_agent(trained_agent, env, num_episodes=100)
    print(f"SAC Average Reward: {avg_reward}")

def multi_agent_example():
    print("Multi-Agent RL Example")
    env = create_environment('CartPole-v1')
    multi_agent = MultiAgentRL(num_agents=3, state_size=env.observation_space.shape[0], action_size=env.action_space.n)
    
    for episode in range(200):
        states, _ = env.reset()
        states = [multi_agent.process_state(states) for _ in range(3)]
        total_reward = 0
        for _ in range(500):
            actions = multi_agent.act(states)
            next_states, rewards, dones, _, _ = env.step(actions[0])  # Use action from first agent
            next_states = [multi_agent.process_state(next_states) for _ in range(3)]
            multi_agent.train(states, actions, [rewards]*3, next_states, [dones]*3)
            states = next_states
            total_reward += rewards
            if dones:
                break
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

def hierarchical_rl_example():
    print("Hierarchical RL Example")
    env = create_environment('MountainCar-v0')
    agent = HierarchicalRL(env.observation_space.shape[0], env.action_space.n, num_options=4)
    
    for episode in range(300):
        state, _ = env.reset()
        state = agent.process_state(state)
        total_reward = 0
        for _ in range(1000):
            action, option = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = agent.process_state(next_state)
            agent.train(state, action, reward, next_state, done, option)
            state = next_state
            total_reward += reward
            if done:
                break
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

def curiosity_driven_rl_example():
    print("Curiosity-Driven RL Example")
    env = create_environment('MountainCar-v0')
    agent = CuriosityDrivenRL(env.observation_space.shape[0], env.action_space.n)
    
    for episode in range(300):
        state, _ = env.reset()
        state = agent.process_state(state)
        total_reward = 0
        for _ in range(1000):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = agent.process_state(next_state)
            agent.train(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

def meta_learning_example():
    print("Meta-Learning Example")
    env = create_environment('CartPole-v1')
    agent = MetaLearningAgent(env.observation_space.shape[0], env.action_space.n)
    
    # Generate episodes for meta-training
    episodes = []
    for _ in range(50):
        episode = []
        state, _ = env.reset()
        for _ in range(200):
            action = env.action_space.sample()
            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward, next_state, done))
            if done:
                break
            state = next_state
        episodes.append(episode)
    
    # Meta-train the agent
    agent.meta_train(episodes)
    
    # Test the meta-learned policy
    for episode in range(10):
        state, _ = env.reset()
        total_reward = 0
        adapted_q_values = agent.adapt(episodes[0])  # Adapt using the first episode
        for _ in range(200):
            action = agent.act(state, adapted_q_values)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break
            state = next_state
        print(f"Episode {episode}, Total Reward: {total_reward}")

if __name__ == "__main__":
    dqn_example()
    a2c_example()
    ppo_example()
    sac_example()
    multi_agent_example()
    hierarchical_rl_example()
    curiosity_driven_rl_example()
    meta_learning_example()