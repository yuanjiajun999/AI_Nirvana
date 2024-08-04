# Intelligent Agent 模块

## 概述

Intelligent Agent 模块实现了一个基于深度 Q 网络 (DQN) 的智能代理系统。这个系统能够通过与环境的交互来学习和优化其行为。

## 主要组件

### IntelligentAgent (抽象基类)

这是所有智能代理的基类，定义了智能代理应该具有的基本方法。

#### 方法：

- `perceive(observation)`: 感知环境状态
- `decide(state)`: 根据当前状态决定行动
- `act(state)`: 执行决定的行动
- `remember(state, action, reward, next_state, done)`: 记忆经验
- `replay(batch_size)`: 回放经验进行学习
- `load(name)`: 加载模型权重
- `save(name)`: 保存模型权重
- `run(environment, episodes, max_steps)`: 在给定环境中运行智能代理

### DQNAgent

这是 `IntelligentAgent` 的具体实现，使用深度 Q 网络来学习最优策略。

#### 额外方法：

- `_build_model()`: 构建神经网络模型

## 使用示例

```python
from src.core.intelligent_agent import DQNAgent

# 初始化智能代理
agent = DQNAgent(state_size=4, action_size=2)

# 假设我们有一个 environment 对象，它有 reset() 和 step() 方法

for episode in range(num_episodes):
    state = environment.reset()
    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, _ = environment.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    
    # 训练智能代理
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
注意事项

确保在使用 DQNAgent 之前正确设置了 state_size 和 action_size。
replay 方法需要足够的记忆样本才能开始训练，建议在记忆大小超过批次大小时调用。
可以通过调整 epsilon、epsilon_decay 和 epsilon_min 参数来控制探索与利用的平衡。

未来改进

实现更多种类的强化学习算法，如 DDPG、A3C 等。
添加优先经验回放 (Prioritized Experience Replay) 功能。
实现多智能体系统的支持。

