# examples/reinforcement_learning_example.py

from src.core.reinforcement_learning import ReinforcementLearningAgent
import numpy as np

class SimpleEnvironment:
    def __init__(self):
        self.state = 0

    def step(self, action):
        if action == 0:  # 左移
            self.state = max(0, self.state - 1)
        elif action == 1:  # 右移
            self.state = min(5, self.state + 1)
        
        reward = 1 if self.state == 5 else 0
        done = self.state == 5
        return self.state, reward, done

def main():
    env = SimpleEnvironment()
    agent = ReinforcementLearningAgent(state_size=6, action_size=2)

    n_episodes = 1000
    for episode in range(n_episodes):
        state = env.state
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    print("Training completed.")
    
    # 测试学习到的策略
    state = 0
    steps = 0
    while state != 5:
        action = agent.act(state)
        state, _, _ = env.step(action)
        steps += 1
    print(f"Steps to reach goal: {steps}")

if __name__ == "__main__":
    main()