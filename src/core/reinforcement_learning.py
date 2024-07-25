# core/reinforcement_learning.py

import numpy as np
import tensorflow as tf


class ReinforcementLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    24, activation="relu", input_shape=(self.state_size,)
                ),
                tf.keras.layers.Dense(24, activation="relu"),
                tf.keras.layers.Dense(self.action_size, activation="linear"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse"
        )
        return model

    def act(self, state):
        state = np.reshape(state, [1, self.state_size])
        action_values = self.model.predict(state)
        return np.argmax(action_values[0])

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + 0.95 * np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
