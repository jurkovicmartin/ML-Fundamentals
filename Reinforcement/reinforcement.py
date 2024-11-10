import numpy as np
# Only for notations
from maze import Maze

import random

class Agent:
    def __init__(self, env_path, actions: list):
        self.environment = Maze(env_path)
        self.actions = actions

        self.q_table = {action: np.zeros(self.environment.shape) for action in actions}

    
    def learn(self, epsilon: float, alpha: float, gamma: float, epsilon_min: float, episodes: int, max_steps: int):
        self.epsilon = epsilon

        for episode in range(episodes):
            self.environment.reset()
            score = 0
            steps_count = 0

            while steps_count < max_steps:
                # Current based on iteration
                current_position = self.environment.current_pos

                next_step = self._choose_action(current_position)

                move_result = self.environment.move(next_step)
                # New position
                next_position = self.environment.current_pos

                ### REWARDING SYSTEM
                # Hitting a wall
                if not move_result:
                    reward = -10
                    finish = False
                # Regular step
                elif move_result and not self.environment.is_finished():
                    reward = -1
                    finish = False
                # Reaching finish
                elif move_result and self.environment.is_finished():
                    reward = 100
                    finish = True
                else:
                    raise Exception("Rewarding error")
                
                current_Q = self.q_table[next_step][current_position]

                next_Q = -float("inf")
                for action in self.actions:
                    if self.q_table[action][next_position] > next_Q:
                        next_Q = self.q_table[action][next_position]

                self.q_table[next_step][current_position] = current_Q + \
                            alpha * (reward + gamma * next_Q - current_Q)
                
                if finish:
                    break

                score += reward
                steps_count += 1

            print(f"Episode: {episode + 1}, Score: {score}")

            # if finish:
            #     break

            if self.epsilon > epsilon_min:
                self.epsilon -= (epsilon - epsilon_min) / episodes


    def _choose_action(self, position: tuple) -> str:
        # Exploration
        if np.random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        # Exploitation
        else:
            best_Q = -float("inf")
            best_action = None
            for action in self.actions:
                if self.q_table[action][position] > best_Q:
                    best_Q = self.q_table[action][position]
                    best_action = action
            return best_action