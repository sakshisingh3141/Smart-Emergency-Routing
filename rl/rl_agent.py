import numpy as np
import random

class RLAgent:

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, neighbors):

        if random.random() < self.epsilon:
            return random.choice(neighbors)

        q_values = [self.get_q(state, n) for n in neighbors]
        return neighbors[np.argmax(q_values)]

    def update(self, state, action, reward, next_state, next_neighbors):

        max_q_next = 0
        if next_neighbors:
            max_q_next = max([self.get_q(next_state, n) for n in next_neighbors])

        current_q = self.get_q(state, action)

        new_q = current_q + self.alpha * (
            reward + self.gamma * max_q_next - current_q
        )

        self.q_table[(state, action)] = new_q