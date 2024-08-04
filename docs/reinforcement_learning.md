# Reinforcement Learning Module Documentation

## Overview

This module provides a comprehensive set of reinforcement learning algorithms and utilities. It includes implementations of various popular RL algorithms, multi-agent systems, hierarchical RL, curiosity-driven learning, and meta-learning approaches.

## Algorithms

### DQNAgent (Deep Q-Network)

The DQN algorithm combines Q-learning with deep neural networks to handle high-dimensional state spaces.

#### Key Features:
- Experience replay
- Target network for stability
- Epsilon-greedy exploration

#### Usage:
```python
agent = DQNAgent(state_size, action_size)
```

### A2CAgent (Advantage Actor-Critic)

A2C is an on-policy algorithm that learns both a policy and a value function.

#### Key Features:
- Separate actor and critic networks
- Advantage function for reduced variance

#### Usage:
```python
agent = A2CAgent(state_size, action_size)
```

### PPOAgent (Proximal Policy Optimization)

PPO is an on-policy algorithm that aims to strike a balance between ease of implementation, sample complexity, and ease of tuning.

#### Key Features:
- Clipped surrogate objective
- Multiple epochs of optimization on each batch of data

#### Usage:
```python
agent = PPOAgent(state_size, action_size)
```

### SACAgent (Soft Actor-Critic)

SAC is an off-policy algorithm that optimizes a stochastic policy in an off-policy way, forming a bridge between stochastic policy optimization and DDPG-style approaches.

#### Key Features:
- Entropy regularization
- Two Q-functions to mitigate positive bias

#### Usage:
```python
agent = SACAgent(state_size, action_size)
```

### TD3Agent (Twin Delayed DDPG)

TD3 is an algorithm which addresses function approximation error in Actor-Critic methods.

#### Key Features:
- Clipped double Q-learning
- Delayed policy updates
- Target policy smoothing

#### Usage:
```python
agent = TD3Agent(state_size, action_size)
```

### DDPGAgent (Deep Deterministic Policy Gradient)

DDPG is an algorithm which concurrently learns a Q-function and a policy.

#### Key Features:
- Off-policy
- Actor-Critic
- Continuous action spaces

#### Usage:
```python
agent = DDPGAgent(state_size, action_size)
```

## Advanced Techniques

### MultiAgentRL

This class allows for training multiple agents in a shared environment.

#### Usage:
```python
multi_agent = MultiAgentRL(num_agents, state_size, action_size, agent_type='DQN')
```

### HierarchicalRL

Implements a hierarchical reinforcement learning approach with options.

#### Usage:
```python
agent = HierarchicalRL(state_size, action_size, num_options=4)
```

### CuriosityDrivenRL

Implements curiosity-driven exploration to help with sparse reward environments.

#### Usage:
```python
agent = CuriosityDrivenRL(state_size, action_size)
```

### MetaLearningAgent

Implements a meta-learning approach to quickly adapt to new tasks.

#### Usage:
```python
agent = MetaLearningAgent(state_size, action_size)
```

## Utility Functions

### create_environment(env_name: str) -> gym.Env
Creates and returns a Gym environment.

### train_agent(agent: BaseAgent, env: gym.Env, num_episodes: int, max_steps_per_episode: int) -> BaseAgent
Trains an agent in the given environment.

### evaluate_agent(agent: BaseAgent, env: gym.Env, num_episodes: int) -> Tuple[float, float]
Evaluates an agent's performance in the given environment.

### plot_learning_curve(rewards: List[float], window_size: int = 100)
Plots the learning curve of an agent.

### save_agent(agent: BaseAgent, filename: str)
Saves an agent to a file.

### load_agent(filename: str) -> BaseAgent
Loads an agent from a file.

### parallel_train_agents(agents: List[BaseAgent], envs: List[gym.Env], num_episodes: int, max_steps_per_episode: int)
Trains multiple agents in parallel.

## Best Practices

1. Start with simpler algorithms (e.g., DQN) before moving to more complex ones.
2. Always monitor the learning curves to detect issues early.
3. Use appropriate preprocessing for your state space.
4. Experiment with hyperparameters, especially learning rate and network architecture.
5. For continuous action spaces, prefer algorithms like SAC or TD3.
6. When dealing with sparse rewards, consider using CuriosityDrivenRL.
7. For complex tasks that can be broken down, try HierarchicalRL.
8. If you're dealing with multiple similar tasks, MetaLearningAgent might be beneficial.

## Troubleshooting

- If learning is unstable, try reducing the learning rate or increasing the batch size.
- If the agent is not exploring enough, increase the exploration rate (epsilon for DQN, entropy coefficient for SAC).
- For off-policy algorithms, ensure your replay buffer is large enough.
- If using PPO, make sure the clipping epsilon is appropriate for your problem.

## Future Improvements

- Implement Distributional RL algorithms
- Add support for image-based observations
- Implement model-based RL algorithms
- Add more extensive logging and visualization tools