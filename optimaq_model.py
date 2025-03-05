
## Core Q-Learning Implementation

## This module implements the reinforcement learning model for addiction prediction.
## It maintains separate Q-tables for positive and negative rewards and applies
## state-action transition dynamics based on patient behavioral trends.


import numpy as np
import logging
from tqdm import tqdm

class QPredictionModel:
    def __init__(self, states, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2, weight_params=None):
        self.states = states
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.Q_pos = np.zeros((states, actions))
        self.Q_neg = np.zeros((states, actions))
        self.weight_params = weight_params or {'healthy': {}, 'continuous_use': {}, 'relapsing': {}, 'recovering': {}}
        logging.info("QPredictionModel initialized.")
    
    def choose_action(self, state):
        combined_q = self.Q_pos[state] - self.Q_neg[state]
        return np.argmax(combined_q) if np.random.rand() > self.epsilon else np.random.randint(self.actions)
    
    def update(self, state, action, reward, next_state, done):
        pos_reward, neg_reward = max(0, reward), max(0, -reward)
        best_next_action = np.argmax(self.Q_pos[next_state] - self.Q_neg[next_state]) if not done else 0
        self.Q_pos[state, action] += self.lr * (pos_reward + self.gamma * self.Q_pos[next_state, best_next_action] - self.Q_pos[state, action])
        self.Q_neg[state, action] += self.lr * (neg_reward + self.gamma * self.Q_neg[next_state, best_next_action] - self.Q_neg[state, action])
    
    def train(self, X_train, y_train, num_episodes=1000):
        logging.info("Training QPredictionModel...")
        for _ in tqdm(range(num_episodes), desc="Training episodes"):
            state = np.random.choice(len(X_train))
            action = self.choose_action(state)
            reward = y_train[state]  # Simulated reward
            next_state = (state + 1) % len(X_train)
            self.update(state, action, reward, next_state, False)
        logging.info("Training complete.")
    
    def predict_batch(self, X):
        return [self.choose_action(i) for i in range(len(X))]
