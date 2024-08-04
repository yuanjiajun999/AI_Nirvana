from src.core.intelligent_agent import DQNAgent
import numpy as np

def main():
    # 初始化 DQNAgent
    state_size = 4
    action_size = 2
    agent = DQNAgent(state_size, action_size)

    # 模拟环境
    class SimpleEnvironment:
        def __init__(self):
            self.state = np.zeros(state_size)
            self.step_count = 0

        def reset(self):
            self.state = np.random.rand(state_size)
            self.step_count = 0
            return self.state

        def step(self, action):
            self.state = np.random.rand(state_size)
            self.step_count += 1
            reward = np.random.rand()
            done = self.step_count >= 10
            return self.state, reward, done, {}

    env = SimpleEnvironment()

    # 运行智能代理
    episodes = 5
    max_steps = 20

    for episode in range(episodes):
        state = env.reset()
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        
        print(f"Episode {episode + 1}/{episodes} completed.")
        
        # 训练智能代理
        if len(agent.memory) > 32:
            agent.replay(32)

    print("Training completed.")

if __name__ == "__main__":
    main()