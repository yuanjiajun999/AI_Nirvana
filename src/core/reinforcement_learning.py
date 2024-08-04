# E:\AI_Nirvana-1\src\core\reinforcement_learning.py  

from multiprocessing import Pool  
import numpy as np  
import tensorflow as tf  
import random  
from typing import Tuple, Union  
from collections import deque  
from keras.layers import Dense  
from keras.optimizers import Adam  
from keras.models import Sequential  
import gym  
from typing import List, Tuple  
from abc import ABC, abstractmethod  
import matplotlib.pyplot as plt  
import pickle
from tensorflow import keras  
from tensorflow.keras import layers

def process_state(state):
    if isinstance(state, (list, tuple, np.ndarray)):
        return np.array(state, dtype=np.float32).reshape(1, -1)
    elif isinstance(state, dict):
        return np.array(list(state.values()), dtype=np.float32).reshape(1, -1)
    else:
        return np.array([state], dtype=np.float32).reshape(1, -1)
    
class BaseAgent(ABC):  
    def __init__(self):  
        pass  

    @abstractmethod  
    def act(self, state):  
        pass  

    @abstractmethod  
    def train(self, state, action, reward, next_state, done):  
        pass  

    def process_state(self, state):  
        return process_state(state)  

class DQNAgent(BaseAgent):  
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=10000, batch_size=32, update_target_frequency=100):  
        super().__init__()  
        self.state_size = state_size  
        self.action_size = action_size  
        self.memory = deque(maxlen=memory_size)  
        self.gamma = gamma  
        self.epsilon = epsilon  
        self.epsilon_decay = epsilon_decay  
        self.epsilon_min = epsilon_min  
        self.learning_rate = learning_rate  
        self.batch_size = batch_size  
        self.update_target_frequency = update_target_frequency  
        self.model = self._build_model()  
        self.target_model = self._build_model()  
        self.update_target_model()  
        self.step_count = 0  
        self.train_count = 0  # 添加 train_count 属性  

    def _build_model(self):  
        model = Sequential([  
            Dense(64, activation='relu', input_shape=(self.state_size,)),  
            Dense(64, activation='relu'),  
            Dense(self.action_size, activation='linear')  
        ])  
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')  
        return model  

    def update_target_model(self):  
        self.target_model.set_weights(self.model.get_weights())  

    def remember(self, state, action, reward, next_state, done):  
        self.memory.append((state, action, reward, next_state, done))  

    def act(self, state):  
        state = self.process_state(state).reshape(1, -1)  
        if np.random.rand() <= self.epsilon:  
            return random.randrange(self.action_size)  
        act_values = self.model.predict(state)  
        return np.argmax(act_values[0])  

    def train(self, state, action, reward, next_state, done):  
        state = self.process_state(state).reshape(1, -1)  
        next_state = self.process_state(next_state).reshape(1, -1)  

        self.remember(state, action, reward, next_state, done)  

        if len(self.memory) < self.batch_size:  
            loss = 0  
        else:  
            loss = self.replay(self.batch_size)  

        loss = loss if loss is not None else 0  

        self.step_count += 1  
        if self.step_count % self.update_target_frequency == 0:  
            self.update_target_model()  

        # Epsilon decay - 每次训练都衰减  
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)  

        self.train_count += 1  # 增加训练计数  

        return loss  
    
    def replay(self, batch_size):  
        if len(self.memory) < batch_size:  
            return None  

        minibatch = random.sample(self.memory, batch_size)  
        states = np.array([self.process_state(state).reshape(1, -1)[0] for state, _, _, _, _ in minibatch])  
        targets = self.model.predict(states)  

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):  
            if done:  
                targets[i][action] = reward  
            else:  
                next_state = self.process_state(next_state).reshape(1, -1)  
                Q_future = np.max(self.target_model.predict(next_state)[0])  
                targets[i][action] = reward + self.gamma * Q_future  

        history = self.model.fit(states, targets, epochs=1, verbose=0)  

        return history.history['loss'][0] if 'loss' in history.history else None  
   
    @staticmethod  
    def process_state(state):  
        """  
        Convert the state to a numpy array of floats in a consistent manner.  
        """  
        if isinstance(state, dict):  
            # 处理字典输入，将值转换为列表  
            processed = np.array(list(state.values()), dtype=np.float32)  
        elif isinstance(state, (list, tuple, np.ndarray)):  
            processed = np.concatenate([  
                DQNAgent.process_state(s) if isinstance(s, (list, tuple, np.ndarray, dict))  
                else np.array([s], dtype=np.float32) for s in state  
            ])  
        else:  
            processed = np.array([state], dtype=np.float32)  
        
        # 确保输出始终是 (1, n) 的形状  
        return processed.reshape(1, -1)  

    def get_epsilon(self):  
        """  
        Returns the current epsilon value.  
        """  
        return self.epsilon  
    
    def get_train_count(self):  
        """  
        Returns the current train count.  
        """  
        return self.train_count  
    
class A2CAgent(BaseAgent):  
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):  
        super().__init__()  
        self.state_size = state_size  
        self.action_size = action_size  
        self.gamma = gamma  
        self.learning_rate = learning_rate  
        self.actor, self.critic = self._build_model()  
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)  
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)  

    def _build_model(self):  
        input_layer = tf.keras.layers.Input(shape=(self.state_size,))  
        dense1 = tf.keras.layers.Dense(64, activation='relu')(input_layer)  
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)  
        
        actor_output = tf.keras.layers.Dense(self.action_size, activation='softmax')(dense2)  
        critic_output = tf.keras.layers.Dense(1)(dense2)  
        
        actor = tf.keras.Model(inputs=input_layer, outputs=actor_output)  
        critic = tf.keras.Model(inputs=input_layer, outputs=critic_output)  
        
        return actor, critic  

    def act(self, state):  
        state = self.process_state(state)  
        if isinstance(state, tuple):  
            state = state[0]  # 提取实际的状态数组  
        if len(state.shape) == 1:  
            state = np.reshape(state, [1, -1])  
        probs = self.actor(state).numpy()[0]  
        return np.random.choice(self.action_size, p=probs)  

    def train(self, state, action, reward, next_state, done):  
        state = self.process_state(state)  
        next_state = self.process_state(next_state)  
        # 处理单个状态或批量状态  
        if isinstance(state, (list, tuple, np.ndarray)):  
            state = np.array([s[0] if isinstance(s, tuple) else s for s in state])  
        elif isinstance(state, tuple):  
            state = state[0]  

        if isinstance(next_state, (list, tuple, np.ndarray)):  
            next_state = np.array([s[0] if isinstance(s, tuple) else s for s in next_state])  
        elif isinstance(next_state, tuple):  
            next_state = next_state[0]  

        state = np.reshape(state, [1, self.state_size])  
        next_state = np.reshape(next_state, [1, self.state_size])  

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:  
            probs = self.actor(state)  
            value = self.critic(state)  
            next_value = self.critic(next_state)  
    
            advantage = reward + (1 - done) * self.gamma * next_value - value  
            critic_loss = tf.reduce_mean(tf.square(advantage))  
    
            action_onehot = tf.one_hot(action, self.action_size)  
            log_prob = tf.math.log(tf.reduce_sum(probs * action_onehot))  
            actor_loss = -log_prob * tf.stop_gradient(advantage)  

        actor_grads = tape1.gradient(actor_loss, self.actor.trainable_variables)  
        critic_grads = tape2.gradient(critic_loss, self.critic.trainable_variables)  

        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))  
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))  

        # 将 actor_loss 和 critic_loss 转换为标量值  
        actor_loss_scalar = float(actor_loss.numpy())  
        critic_loss_scalar = float(critic_loss.numpy())  

        # 返回单个损失值（平均损失）  
        return (actor_loss_scalar + critic_loss_scalar) / 2

    def process_state(self, state):  
        if len(state.shape) == 3:  
            return state.squeeze(0)  # Remove the batch dimension if present  
        return state
    
class PPOAgent(BaseAgent):  
    def __init__(self, state_size, action_size, learning_rate=0.0003, gamma=0.99, clip_ratio=0.2, epochs=10):  
        super().__init__()  
        if isinstance(state_size, (list, tuple)):  
            self.state_size = state_size[0]  # 使用第一个维度  
        else:  
            self.state_size = state_size  
        self.action_size = action_size  
        self.gamma = gamma  
        self.learning_rate = learning_rate  
        self.clip_ratio = clip_ratio  
        self.epochs = epochs  
        self.actor, self.critic = self._build_model()  
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)  

    def _build_model(self):  
        input_layer = tf.keras.layers.Input(shape=(self.state_size,))  
        dense1 = tf.keras.layers.Dense(64, activation='tanh')(input_layer)  
        dense2 = tf.keras.layers.Dense(64, activation='tanh')(dense1)  
        
        actor_output = tf.keras.layers.Dense(self.action_size, activation='softmax')(dense2)  
        critic_output = tf.keras.layers.Dense(1)(dense2)  
        
        actor = tf.keras.Model(inputs=input_layer, outputs=actor_output)  
        critic = tf.keras.Model(inputs=input_layer, outputs=critic_output)  
        
        return actor, critic  

    def act(self, state):  
        state = self.process_state(state)
        state = self.process_state(state).reshape(1, -1)  
        probs = self.actor(state).numpy()[0]  
        return np.random.choice(self.action_size, p=probs)  

    def train(self, states, actions, rewards, next_states, dones):  
        states = np.array([self.process_state(s) for s in states])  
        next_states = np.array([self.process_state(s) for s in next_states])  

        # Ensure states and next_states are 2D arrays  
        if states.ndim == 1:  
            states = np.expand_dims(states, axis=0)  
        if next_states.ndim == 1:  
            next_states = np.expand_dims(next_states, axis=0)  

        # Debugging Information  
        print(f"States shape: {states.shape}")  
        print(f"Next States shape: {next_states.shape}")  

        # 确保 actions 是一个数组  
        actions = np.atleast_1d(actions)  
        rewards = np.array(rewards)  
        dones = np.array(dones)  

        states = tf.convert_to_tensor(states, dtype=tf.float32)  
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)  
    
        old_probs = self.actor(states).numpy()  
        old_log_probs = np.log(old_probs[np.arange(len(actions)), actions])  

        values = self.critic(states).numpy().flatten()  
        next_values = self.critic(next_states).numpy().flatten()  

        advantages = rewards + self.gamma * next_values * (1 - dones) - values  
        returns = rewards + self.gamma * next_values * (1 - dones)  

        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)  

        actor_losses = []  
        critic_losses = []  

        for _ in range(self.epochs):  
            with tf.GradientTape() as tape:  
                new_probs = self.actor(states)  
                new_log_probs = tf.math.log(tf.reduce_sum(new_probs * tf.one_hot(actions, self.action_size), axis=1))  
                ratio = tf.exp(new_log_probs - old_log_probs)  
                clipped_advantages = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages  
                actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_advantages))  

                critic_values = tf.squeeze(self.critic(states))  
                critic_loss = tf.reduce_mean(tf.square(returns - critic_values))  

                total_loss = actor_loss + 0.5 * critic_loss  

            grads = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)  
            self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))  

            actor_losses.append(actor_loss.numpy())  
            critic_losses.append(critic_loss.numpy())  

        # 返回平均损失  
        return np.mean(actor_losses + critic_losses)

    def process_state(self, state):  
        if len(state.shape) == 3:  
            return state.squeeze(0)  # Remove the batch dimension if present  
        return state 

class SACAgent(BaseAgent):  
    def __init__(self, state_size, action_size, learning_rate=0.0003, gamma=0.99, tau=0.005, alpha=0.2):  
        super().__init__()  
        self.state_size = state_size  
        self.action_size = action_size  
        self.gamma = gamma  
        self.tau = tau  
        self.alpha = alpha  
        self.learning_rate = learning_rate  
        
        self.actor = self._build_actor()  
        self.critic1, self.critic2 = self._build_critics()  
        self.target_critic1, self.target_critic2 = self._build_critics()  
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)  
        self.critic1_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)  
        self.critic2_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)  
        
        self.update_target_networks(tau=1.0)  

    def _build_actor(self):  
        inputs = layers.Input(shape=(self.state_size,))  
        x = layers.Dense(64, activation='relu')(inputs)  
        x = layers.Dense(64, activation='relu')(x)  
        outputs = layers.Dense(self.action_size * 2, activation='linear')(x)  
        return keras.Model(inputs, outputs) 

    def _build_critics(self):  
        critic = tf.keras.Sequential([  
            tf.keras.layers.Dense(256, activation='relu', input_shape=(self.state_size + self.action_size,)),  
            tf.keras.layers.Dense(256, activation='relu'),  
            tf.keras.layers.Dense(1)  
        ])  
        return critic, tf.keras.models.clone_model(critic)  

    def act(self, state):
        state = self.process_state(state)
        if isinstance(state, tuple):
            state = state[0]  # 提取实际的状态数组
        if len(state.shape) == 1:
            state = np.reshape(state, [1, -1])
        action, _ = self.sample_action(state)
        return tf.tanh(action).numpy()[0] 

    def train(self, state, action, reward, next_state, done):  
        state = self.process_state(state)  
        next_state = self.process_state(next_state)  
        # 处理单个状态或批量状态  
        if isinstance(state, (list, tuple, np.ndarray)):  
            state = np.array([s[0] if isinstance(s, tuple) else s for s in state])  
        elif isinstance(state, tuple):  
            state = state[0]  

        if isinstance(next_state, (list, tuple, np.ndarray)):  
            next_state = np.array([s[0] if isinstance(s, tuple) else s for s in next_state])  
        elif isinstance(next_state, tuple):  
            next_state = next_state[0]  
    
        state = np.reshape(state, [1, self.state_size])  
        next_state = np.reshape(next_state, [1, self.state_size])  
        action = np.reshape(action, [1, self.action_size])  
        reward = np.reshape(reward, [1, 1])  
        done = np.reshape(done, [1, 1])  

        with tf.GradientTape(persistent=True) as tape:  
            next_action, next_log_pi = self.sample_action(next_state)  
            target_q1 = self.target_critic1(tf.concat([next_state, next_action], axis=-1))  
            target_q2 = self.target_critic2(tf.concat([next_state, next_action], axis=-1))  
            target_q = tf.minimum(target_q1, target_q2) - self.alpha * next_log_pi  
            target_q = reward + (1 - done) * self.gamma * target_q  

            current_q1 = self.critic1(tf.concat([state, action], axis=-1))  
            current_q2 = self.critic2(tf.concat([state, action], axis=-1))  
            critic1_loss = tf.reduce_mean(tf.square(current_q1 - target_q))  
            critic2_loss = tf.reduce_mean(tf.square(current_q2 - target_q))  

            new_action, log_pi = self.sample_action(state)  
            q1 = self.critic1(tf.concat([state, new_action], axis=-1))  
            q2 = self.critic2(tf.concat([state, new_action], axis=-1))  
            q = tf.minimum(q1, q2)  
            actor_loss = tf.reduce_mean(self.alpha * log_pi - q)  

        critic1_grad = tape.gradient(critic1_loss, self.critic1.trainable_variables)  
        critic2_grad = tape.gradient(critic2_loss, self.critic2.trainable_variables)  
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)  

        self.critic1_optimizer.apply_gradients(zip(critic1_grad, self.critic1.trainable_variables))  
        self.critic2_optimizer.apply_gradients(zip(critic2_grad, self.critic2.trainable_variables))  
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))  

        self.update_target_networks()  

        # 将损失转换为标量值并取绝对值  
        actor_loss = abs(float(actor_loss.numpy()))  
        critic1_loss = abs(float(critic1_loss.numpy()))  
        critic2_loss = abs(float(critic2_loss.numpy()))  

        # 返回单个损失值（平均损失的绝对值）  
        return (actor_loss + critic1_loss + critic2_loss) / 3  
   
    def sample_action(self, state):  
       # 确保状态的形状正确
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)  # 添加批次维度
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        
        mean, log_std = tf.split(self.actor(state), 2, axis=-1)
        std = tf.exp(log_std)
        z = mean + tf.random.normal(tf.shape(mean)) * std
        action = tf.tanh(z)
        log_pi = tf.reduce_sum(
            tf.math.log(1 - tf.square(action) + 1e-6) - log_std - 0.5 * np.log(2 * np.pi) - 0.5 * tf.square(z),
            axis=1, keepdims=True
        )
        return action, log_pi

    def update_target_networks(self, tau=None):  
        if tau is None:  
            tau = self.tau  
        for target_var, var in zip(self.target_critic1.variables, self.critic1.variables):  
            target_var.assign(tau * var + (1 - tau) * target_var)  
        for target_var, var in zip(self.target_critic2.variables, self.critic2.variables):  
            target_var.assign(tau * var + (1 - tau) * target_var)

    def process_state(self, state):  
        if len(state.shape) == 3:  
            return state.squeeze(0)  # Remove the batch dimension if present  
        return state

class TD3Agent(BaseAgent):  
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):  
        super().__init__()  
        self.state_size = state_size  
        self.action_size = action_size  
        self.gamma = gamma  
        self.tau = tau  
        self.policy_noise = policy_noise  
        self.noise_clip = noise_clip  
        self.policy_freq = policy_freq  
        
        self.actor = self._build_actor()  
        self.actor_target = self._build_actor()  
        self.critic1, self.critic2 = self._build_critics()  
        self.critic1_target, self.critic2_target = self._build_critics()  
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  
        
        self.update_target_networks(tau=1.0)  
        self.total_it = 0  

    def _build_actor(self):  
        inputs = layers.Input(shape=(self.state_size,))  
        x = layers.Dense(64, activation='relu')(inputs)  
        x = layers.Dense(64, activation='relu')(x)  
        outputs = layers.Dense(self.action_size, activation='tanh')(x)  
        return keras.Model(inputs, outputs) 

    def _build_critics(self):  
        critic = tf.keras.Sequential([  
            tf.keras.layers.Dense(256, activation='relu', input_shape=(self.state_size + self.action_size,)),  
            tf.keras.layers.Dense(256, activation='relu'),  
            tf.keras.layers.Dense(1)  
        ])  
        return critic, tf.keras.models.clone_model(critic)  

    def act(self, state):  
        state = self.process_state(state)  
        if isinstance(state, tuple):  
            state = state[0]  # 提取实际的状态数组  
        if len(state.shape) == 1:  
            state = np.reshape(state, [1, -1])  
        return self.actor(state).numpy()[0]  

    def train(self, state, action, reward, next_state, done):  
        state = self.process_state(state)  
        next_state = self.process_state(next_state)  
        # 处理单个状态或批量状态  
        if isinstance(state, (list, tuple, np.ndarray)):  
            state = np.array([s[0] if isinstance(s, tuple) else s for s in state])  
        elif isinstance(state, tuple):  
            state = state[0]  

        if isinstance(next_state, (list, tuple, np.ndarray)):  
            next_state = np.array([s[0] if isinstance(s, tuple) else s for s in next_state])  
        elif isinstance(next_state, tuple):  
            next_state = next_state[0]  
    
        self.total_it += 1  
        state = np.reshape(state, [1, self.state_size])  
        next_state = np.reshape(next_state, [1, self.state_size])  
        action = np.reshape(action, [1, self.action_size])  
        reward = np.reshape(reward, [1, 1])  
        done = np.reshape(done, [1, 1])  

        with tf.GradientTape(persistent=True) as tape:  
            noise = tf.clip_by_value(tf.random.normal(tf.shape(action), stddev=self.policy_noise), -self.noise_clip, self.noise_clip)  
            next_action = tf.clip_by_value(self.actor_target(next_state) + noise, -1, 1)  

            target_q1 = self.critic1_target(tf.concat([next_state, next_action], axis=1))  
            target_q2 = self.critic2_target(tf.concat([next_state, next_action], axis=1))  
            target_q = tf.minimum(target_q1, target_q2)  
            target_q = reward + (1 - done) * self.gamma * target_q  

            current_q1 = self.critic1(tf.concat([state, action], axis=1))  
            current_q2 = self.critic2(tf.concat([state, action], axis=1))  

            critic1_loss = tf.reduce_mean(tf.square(current_q1 - target_q))  
            critic2_loss = tf.reduce_mean(tf.square(current_q2 - target_q))  
            critic_loss = critic1_loss + critic2_loss  

        critic_grad = tape.gradient(critic_loss, self.critic1.trainable_variables + self.critic2.trainable_variables)  
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic1.trainable_variables + self.critic2.trainable_variables))  

        actor_loss = None  
        if self.total_it % self.policy_freq == 0:  
            with tf.GradientTape() as tape:  
                actor_loss = -tf.reduce_mean(self.critic1(tf.concat([state, self.actor(state)], axis=1)))  

            actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)  
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))  

            self.update_target_networks()  

        # 将损失转换为标量值并取绝对值  
        critic_loss = abs(float(critic_loss.numpy()))  
        if actor_loss is not None:  
            actor_loss = abs(float(actor_loss.numpy()))  
            return (critic_loss + actor_loss) / 2  
        else:  
            return critic_loss  
    
    def update_target_networks(self, tau=None):  
        if tau is None:  
            tau = self.tau  
        for target_var, var in zip(self.actor_target.variables, self.actor.variables):  
            target_var.assign(tau * var + (1 - tau) * target_var)  
        for target_var, var in zip(self.critic1_target.variables, self.critic1.variables):  
            target_var.assign(tau * var + (1 - tau) * target_var)  
        for target_var, var in zip(self.critic2_target.variables, self.critic2.variables):  
            target_var.assign(tau * var + (1 - tau) * target_var)  

    def process_state(self, state):  
        if len(state.shape) == 3:  
            return state.squeeze(0)  # Remove the batch dimension if present  
        return state

class DDPGAgent(BaseAgent):  
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, tau=0.001):  
        super().__init__()  
        self.state_size = state_size  
        self.action_size = action_size  
        self.gamma = gamma  
        self.tau = tau  
        
        self.actor = self._build_actor()  
        self.actor_target = self._build_actor()  
        self.critic = self._build_critic()  
        self.critic_target = self._build_critic()  
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  
        
        self.update_target_networks(tau=1.0)  

    def _build_actor(self):  
        model = tf.keras.Sequential([  
            tf.keras.layers.Dense(400, activation='relu', input_shape=(self.state_size,)),  
            tf.keras.layers.Dense(300, activation='relu'),  
            tf.keras.layers.Dense(self.action_size, activation='tanh')  
        ])  
        return model  

    def _build_critic(self):  
        state_input = tf.keras.layers.Input(shape=(self.state_size,))  
        action_input = tf.keras.layers.Input(shape=(self.action_size,))  
        x = tf.keras.layers.Concatenate()([state_input, action_input])  
        x = tf.keras.layers.Dense(400, activation='relu')(x)  
        x = tf.keras.layers.Dense(300, activation='relu')(x)  
        x = tf.keras.layers.Dense(1)(x)  
        model = tf.keras.Model(inputs=[state_input, action_input], outputs=x)  
        return model  

    def act(self, state):  
        state = self.process_state(state)  
        if isinstance(state, tuple):  
            state = state[0]  # 提取实际的状态数组  
        state = state.reshape(1, -1)  
        return self.actor(state).numpy()[0]  

    def train(self, state, action, reward, next_state, done):  
        state = self.process_state(state)  
        next_state = self.process_state(next_state)  
        if isinstance(state, tuple):  
            state = state[0]  
        if isinstance(next_state, tuple):  
            next_state = next_state[0]  
        state = state.reshape(1, -1)  
        next_state = next_state.reshape(1, -1)  
        action = np.reshape(action, [1, self.action_size])  
        reward = np.reshape(reward, [1, 1])  
        done = np.reshape(done, [1, 1])  

        with tf.GradientTape() as critic_tape:  
            target_actions = self.actor_target(next_state)  
            target_q = self.critic_target([tf.convert_to_tensor(next_state, dtype=tf.float32),   
                                           tf.convert_to_tensor(target_actions, dtype=tf.float32)])  
            target_q = reward + (1 - done) * self.gamma * target_q  
            current_q = self.critic([tf.convert_to_tensor(state, dtype=tf.float32),   
                                     tf.convert_to_tensor(action, dtype=tf.float32)])  
            critic_loss = tf.reduce_mean(tf.square(target_q - current_q))  
        critic_grad = critic_tape.gradient(critic_loss, self.critic.trainable_variables)  
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))  

        with tf.GradientTape() as actor_tape:  
            actions = self.actor(state)  
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)  
            actions_tensor = tf.convert_to_tensor(actions, dtype=tf.float32)  
            actor_loss = -tf.reduce_mean(self.critic([state_tensor, actions_tensor]))  

        actor_grad = actor_tape.gradient(actor_loss, self.actor.trainable_variables)  
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))  

        self.update_target_networks()  

        # 将 actor_loss 和 critic_loss 转换为标量值  
        actor_loss = float(actor_loss.numpy())  
        critic_loss = float(critic_loss.numpy())  

        # 返回单个损失值（平均损失）  
        return (actor_loss + critic_loss) / 2


    def update_target_networks(self, tau=None):  
        if tau is None:  
            tau = self.tau  
        for target_var, var in zip(self.actor_target.variables, self.actor.variables):  
            target_var.assign(tau * var + (1 - tau) * target_var)  
        for target_var, var in zip(self.critic_target.variables, self.critic.variables):  
            target_var.assign(tau * var + (1 - tau) * target_var)  

class MultiAgentRL:  
    def __init__(self, num_agents: int, state_size: int, action_size: int, agent_type: str = 'DQN'):  
        super().__init__()  
        self.num_agents = num_agents  
        self.state_size = state_size  
        self.action_size = action_size  
        self.agent_type = agent_type  
        self.agents = self._create_agents()  

    def _create_agents(self) -> List[BaseAgent]:  
        agents = []  
        for _ in range(self.num_agents):  
            if self.agent_type == 'DQN':  
                agents.append(DQNAgent(self.state_size, self.action_size))  
            elif self.agent_type == 'A2C':  
                agents.append(A2CAgent(self.state_size, self.action_size))  
            elif self.agent_type == 'PPO':  
                agents.append(PPOAgent(self.state_size, self.action_size))  
            elif self.agent_type == 'SAC':  
                agents.append(SACAgent(self.state_size, self.action_size))  
            elif self.agent_type == 'TD3':  
                agents.append(TD3Agent(self.state_size, self.action_size))  
            elif self.agent_type == 'DDPG':  
                agents.append(DDPGAgent(self.state_size, self.action_size))  
            else:  
                raise ValueError(f"Unsupported agent type: {self.agent_type}")  
        return agents  

    def act(self, states):  
        processed_states = [self.process_state(state).reshape(1, -1) for state in states]  
        return [agent.act(processed_state) for agent, processed_state in zip(self.agents, processed_states)]  

    def train(self, states: List[np.ndarray], actions: List[np.ndarray], rewards: List[float], next_states: List[np.ndarray], dones: List[bool]) -> List[Tuple[float, float]]:
        processed_states = [self.process_state(state) for state in states]
        processed_next_states = [self.process_state(next_state) for next_state in next_states]
        
        results = [agent.train(processed_state, action, reward, processed_next_state, done)
                   for agent, processed_state, action, reward, processed_next_state, done
                   in zip(self.agents, processed_states, actions, rewards, processed_next_states, dones)]
        
        # 确保返回的结果不包含 None 值
        return [(result if result is not None else (0.0, 0.0)) for result in results]

    def process_state(self, state):  
        if isinstance(state, (list, tuple, np.ndarray)):  
            if isinstance(state, tuple):  
                state = state[0]  
            return np.array(state, dtype=np.float32)  
        elif isinstance(state, (int, float)):  
            return np.array([state], dtype=np.float32)  
        else:  
            raise ValueError(f"Unsupported state type: {type(state)}")  

class HierarchicalRL(BaseAgent):  
    def __init__(self, state_size: int, action_size: int, num_options: int = 4):  
        super().__init__()  
        self.state_size = state_size  
        self.action_size = action_size  
        self.num_options = num_options  
        self.meta_controller = DQNAgent(state_size, num_options)  
        self.options = [DQNAgent(state_size, action_size) for _ in range(num_options)]  

    def act(self, state) -> Tuple[int, int]:  
        processed_state = self.process_state(state)  
        if processed_state.ndim == 1:  
            processed_state = processed_state.reshape(1, -1)  
        option = self.meta_controller.act(processed_state)  
        return self.options[option].act(processed_state), option  

    def train(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, option: int):  
        processed_state = self.process_state(state)  
        processed_next_state = self.process_state(next_state)  
        
        if processed_state.ndim == 1:  
            processed_state = processed_state.reshape(1, -1)  
        if processed_next_state.ndim == 1:  
            processed_next_state = processed_next_state.reshape(1, -1)  

        # 训练指定的选项  
        self.options[option].train(processed_state, action, reward, processed_next_state, done)  

        # 如果当前任务已完成，则训练元控制器  
        if done:  
            self.meta_controller.train(processed_state, option, reward, processed_next_state, done)  

    def process_state(self, state):  
        if isinstance(state, (list, tuple)):  
            processed = [self.process_state(s) for s in state]  
            # 添加一个检查，确保所有处理后的状态具有相同的维度  
            shapes = [p.shape for p in processed if p.size > 0]  
            if shapes:  
                target_shape = shapes[0]  
                processed = [p if p.size > 0 else np.zeros(target_shape) for p in processed]  
            return np.concatenate(processed, axis=0)  
        elif isinstance(state, np.ndarray):  
            return state.astype(np.float32)  
        else:  
            return np.array([state], dtype=np.float32)  
        
class CuriosityDrivenRL(BaseAgent):  
    def __init__(self, state_size: int, action_size: int):  
        super().__init__()  
        self.state_size = state_size  
        self.action_size = action_size  
        self.agent = DQNAgent(state_size, action_size)  
        self.forward_model = self._build_forward_model()  
        self.forward_model_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  

    def _build_forward_model(self):  
        model = tf.keras.Sequential([  
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size + self.action_size,)),  
            tf.keras.layers.Dense(64, activation='relu'),  
            tf.keras.layers.Dense(self.state_size)  
        ])  
        return model  

    def act(self, state: np.ndarray) -> int:  
        state = self.process_state(state)  
        if isinstance(state, tuple):  
            state = state[0]  # 提取实际的状态数组  
        if len(state.shape) == 1:  
            state = np.reshape(state, [1, -1])  
        return self.agent.act(state)  

    def train(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):  
        state = self.process_state(state)  
        next_state = self.process_state(next_state)  
        # 处理单个状态或批量状态  
        if isinstance(state, (list, tuple, np.ndarray)):  
            state = np.array([s[0] if isinstance(s, tuple) else s for s in state])  
        elif isinstance(state, tuple):  
            state = state[0]  
    
        if isinstance(next_state, (list, tuple, np.ndarray)):  
            next_state = np.array([s[0] if isinstance(s, tuple) else s for s in next_state])  
        elif isinstance(next_state, tuple):  
            next_state = next_state[0]  
        
        intrinsic_reward = self._compute_intrinsic_reward(state, action, next_state)  
        total_reward = reward + intrinsic_reward  
        self.agent.train(state, action, total_reward, next_state, done)  
        self._train_forward_model(state, action, next_state)  

    def _compute_intrinsic_reward(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:  
        state = np.reshape(state, [1, self.state_size])  
        action = np.reshape(np.eye(self.action_size)[action], [1, self.action_size])  
        next_state = np.reshape(next_state, [1, self.state_size])  
        
        predicted_next_state = self.forward_model(tf.concat([state, action], axis=-1))  
        intrinsic_reward = tf.reduce_mean(tf.square(next_state - predicted_next_state))  
        return intrinsic_reward.numpy()  

    def _train_forward_model(self, state: np.ndarray, action: int, next_state: np.ndarray):  
        state = np.reshape(state, [1, self.state_size])  
        action = np.reshape(np.eye(self.action_size)[action], [1, self.action_size])  
        next_state = np.reshape(next_state, [1, self.state_size])  

        with tf.GradientTape() as tape:  
            predicted_next_state = self.forward_model(tf.concat([state, action], axis=-1))  
            loss = tf.reduce_mean(tf.square(next_state - predicted_next_state))  

        grads = tape.gradient(loss, self.forward_model.trainable_variables)  
        self.forward_model_optimizer.apply_gradients(zip(grads, self.forward_model.trainable_variables))  

class MetaLearningAgent(BaseAgent):  
    def __init__(self, state_size: int, action_size: int, meta_learning_rate: float = 0.001):  
        super().__init__()  
        self.state_size = state_size  
        self.action_size = action_size  
        self.meta_learning_rate = meta_learning_rate  
        self.meta_model = self._build_meta_model()  
        self.meta_optimizer = tf.keras.optimizers.Adam(learning_rate=self.meta_learning_rate)  

    def _build_meta_model(self):  
        model = tf.keras.Sequential([  
            tf.keras.layers.LSTM(64, input_shape=(None, self.state_size + self.action_size + 1)),  
            tf.keras.layers.Dense(64, activation='relu'),  
            tf.keras.layers.Dense(self.action_size, activation='linear')  
        ])  
        return model  

    def meta_train(self, episodes: List[List[Tuple[np.ndarray, int, float, np.ndarray, bool]]]):  
        for episode in episodes:  
            episode_history = []  
            for state, action, reward, next_state, done in episode:  
                state = self.process_state(state).flatten()  
                episode_history.append(np.concatenate([state, np.eye(self.action_size)[action], [float(reward)]]))  
        
            episode_history = np.array(episode_history, dtype=np.float32)  
            episode_history = np.expand_dims(episode_history, axis=0)  
            
            with tf.GradientTape() as tape:  
                q_values = self.meta_model(episode_history)  
                target_values = np.sum(episode_history[:, :, -1], axis=1, keepdims=True)  
                loss = tf.reduce_mean(tf.square(q_values - target_values))  
            
            grads = tape.gradient(loss, self.meta_model.trainable_variables)  
            self.meta_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))  

    def adapt(self, episode: List[Tuple[np.ndarray, int, float, np.ndarray, bool]]):  
        episode_history = []  
        for state, action, reward, next_state, done in episode:  
            state = self.process_state(state).flatten()  
            episode_history.append(np.concatenate([state, np.eye(self.action_size)[action], [reward]]))  
        
        episode_history = np.array(episode_history)  
        episode_history = np.expand_dims(episode_history, axis=0)  
        
        adapted_q_values = self.meta_model(episode_history)  
        return adapted_q_values[0]  
    
    def act(self, state: np.ndarray, adapted_q_values: np.ndarray) -> int:  
        state = self.process_state(state).flatten()  
        inputs = np.concatenate([state, np.zeros(self.action_size + 1)])  
        inputs = np.expand_dims(inputs, axis=0)  
        inputs = np.expand_dims(inputs, axis=0)  # Add another dimension to make it (1, 1, feature_size)  
        q_values = adapted_q_values + self.meta_model(inputs)[0]  
        return np.argmax(q_values)  
    
    def train(self, state, action, reward, next_state, done):  
        # 实现训练逻辑  
        pass  

    def process_state(self, state):  
        if len(state.shape) == 3:  
            return state.squeeze(0).flatten()  
        return state.flatten()

def create_environment(env_name: str) -> gym.Env:  
    return gym.make(env_name)  

def train_agent(agent: BaseAgent, env: gym.Env, num_episodes: int, max_steps_per_episode: int) -> BaseAgent:  
    for episode in range(num_episodes):  
        state, _ = env.reset()  
        state = agent.process_state(state)  
        total_reward = 0  
        for step in range(max_steps_per_episode):  
            action = agent.act(state)  
            next_state, reward, done, truncated, _ = env.step(action)  
            next_state = agent.process_state(next_state)  
            agent.train(state, action, reward, next_state, done)  
            state = next_state  
            total_reward += reward  
            if done or truncated:  
                break  
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")  
    return agent  

def evaluate_agent(agent: BaseAgent, env: gym.Env, num_episodes: int) -> Tuple[float, float]:  
    total_rewards = []  
    for _ in range(num_episodes):  
        state, _ = env.reset()  
        state = agent.process_state(state)  
        episode_reward = 0  
        done = False  
        while not done:  
            action = agent.act(state)  
            next_state, reward, terminated, truncated, _ = env.step(action)  
            done = terminated or truncated  
            episode_reward += reward  
            state = agent.process_state(next_state)  
        total_rewards.append(episode_reward)  
    avg_reward = np.mean(total_rewards)  
    std_reward = np.std(total_rewards)  
    print(f"Evaluation over {num_episodes} episodes:")  
    print(f"Average Reward: {avg_reward:.2f}")  
    print(f"Standard Deviation: {std_reward:.2f}")  
    return avg_reward, std_reward  

def plot_learning_curve(rewards: List[float], window_size: int = 100):  
    plt.figure(figsize=(10, 5))  
    plt.plot(rewards)  
    plt.plot(np.convolve(rewards, np.ones(window_size) / window_size, mode='valid'))  
    plt.title('Learning Curve')  
    plt.xlabel('Episode')  
    plt.ylabel('Reward')  
    plt.legend(['Rewards', f'Moving Average ({window_size} episodes)'])  
    plt.show()  

def save_agent(agent: BaseAgent, filename: str):  
    try:  
        with open(filename, 'wb') as f:  
            pickle.dump(agent, f)  
        print(f"Agent successfully saved to {filename}")  
    except Exception as e:  
        print(f"Error saving agent: {e}")  

def load_agent(filename: str) -> BaseAgent:  
    try:  
        with open(filename, 'rb') as f:  
            agent = pickle.load(f)  
        print(f"Agent successfully loaded from {filename}")  
        return agent  
    except Exception as e:  
        print(f"Error loading agent: {e}")  
        return None  

def train_worker(args):  
    agent, env, num_episodes, max_steps_per_episode = args  
    return train_agent(agent, env, num_episodes, max_steps_per_episode)  

def parallel_train_agents(agents: List[BaseAgent], envs: List[gym.Env], num_episodes: int, max_steps_per_episode: int):  
    with Pool() as pool:  
        args = [(agent, env, num_episodes, max_steps_per_episode) for agent, env in zip(agents, envs)]  
        results = pool.map(train_worker, args)  
    return results 

# 新增一个运行示例的函数  
def run_example():  
    # 创建环境  
    env = create_environment('CartPole-v1')  
    
    # 创建智能体  
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)  
    
    # 训练智能体  
    num_episodes = 1000  
    max_steps_per_episode = 500  
    trained_agent = train_agent(agent, env, num_episodes, max_steps_per_episode)  
    
    # 评估智能体  
    avg_reward, std_reward = evaluate_agent(trained_agent, env, 100)  
    
    # 保存智能体  
    save_agent(trained_agent, 'trained_agent.pkl')  
    
    # 加载智能体  
    loaded_agent = load_agent('trained_agent.pkl')  
    
    # 评估加载的智能体  
    if loaded_agent:  
        avg_reward, std_reward = evaluate_agent(loaded_agent, env, 100)  

if __name__ == "__main__":  
    run_example()  