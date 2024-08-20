# src/core/intelligent_agent.py

import random
import abc
from typing import Any, Dict, List
import numpy as np

class IntelligentAgent(abc.ABC):
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    @abc.abstractmethod
    def _build_model(self):  # pragma: no cover
        """Build the neural network model for the agent."""
        pass

    @abc.abstractmethod
    def perceive(self, observation: Any) -> Any:  # pragma: no cover
        """Perceive the environment and update the agent's internal state."""
        pass

    @abc.abstractmethod
    def decide(self, state: Any) -> Any:  # pragma: no cover
        """Decide on the next action based on the current state."""
        pass
    
    @abc.abstractmethod
    def act(self, state: Any) -> Any:  # pragma: no cover
        """Choose an action based on the current state."""
        pass


    def remember(self, state, action, reward, next_state, done):  # pragma: no cover
        """Store experience in memory."""
        self.memory.append((np.reshape(state, (self.state_size,)),
                            action,
                            reward,
                            np.reshape(next_state, (self.state_size,)),
                            done))

    def replay(self, batch_size):
        """Train on a batch of experiences from memory."""
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """Load the agent's neural network weights."""
        self.model.load_weights(name)

    def save(self, name):
        """Save the agent's neural network weights."""
        if not name.endswith('.weights.h5'):
            name = f"{name}.weights.h5"
        self.model.save_weights(name)

    def run(self, environment, episodes: int, max_steps: int):
        """Run the agent in the given environment for a specified number of episodes."""
        for episode in range(episodes):
            state = environment.reset()
            print("State shape:", np.shape(state))
            print("Expected state size:", self.state_size)
            state = np.reshape(state, [1, self.state_size])
            for step in range(max_steps):
                action = self.act(state)
                next_state, reward, done, _ = environment.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print(f"Episode: {episode+1}/{episodes}, Score: {step+1}")
                    break
            self.replay(32)

class DQNAgent(IntelligentAgent):
    def _build_model(self):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.optimizers import Adam

        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def perceive(self, observation: np.ndarray) -> np.ndarray:
        return np.reshape(observation, [1, self.state_size])

    def decide(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def act(self, state: np.ndarray) -> int:
        return self.decide(state)

    def run(self, env, episodes, max_steps):
        for episode in range(episodes):
            state = env.reset()
            state = self.perceive(state)
            for step in range(max_steps):
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                next_state = self.perceive(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print(f"Episode: {episode+1}/{episodes}, Score: {step+1}")
                    break
            self.replay(32)  # 假设每集之后进行一次经验回放

    