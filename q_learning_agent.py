"""
Q_Learning_Agent.py

This module defines a Q-Learning agent for a discrete action environment.
The agent learns Q-values using temporal difference updates, with support for
epsilon-greedy exploration and epsilon decay. It tracks TD errors for analysis.
"""

import numpy as np
import random
from meals_data import MEALS  # Required to determine the total number of possible actions

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.1):
        """
        Initializes the Q-learning agent with parameters and an empty Q-table.
        """
        self.state_size = state_size
        self.num_total_actions = len(MEALS)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.td_errors = []
        self.q_table = {}

    def _discretize_state(self, state):
        """
        Discretizes continuous state into a tuple for use as a dictionary key.
        """
        return tuple(np.round(state, decimals=1))

    def get_qs(self, state):
        """
        Retrieves Q-values for a given state. Initializes if state is unseen.
        """
        state_key = self._discretize_state(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_total_actions)
        return self.q_table[state_key]

    def select_action(self, state, available_actions):
        """
        Selects an action using epsilon-greedy strategy.
        """
        if np.random.rand() < self.epsilon:
            return random.choice(available_actions)

        qs = self.get_qs(state)
        best_action = max(available_actions, key=lambda a: qs[a])
        return best_action

    def learn(self, current_state, action, reward, next_state, done):
        """
        Performs the Q-learning update for a single step.
        Updates Q-values and tracks TD error.
        """
        current_qs = self.get_qs(current_state)
        next_qs = self.get_qs(next_state)

        td_target = reward + self.gamma * np.max(next_qs) * (1 - int(done))
        td_error = td_target - current_qs[action]
        self.td_errors.append(td_error)

        current_qs[action] += self.lr * td_error

        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
