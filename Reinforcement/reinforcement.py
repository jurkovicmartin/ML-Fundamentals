import numpy as np
from maze import Maze

import random

class Agent:
    def __init__(self, env_path: str, actions: list, display: bool=True):
        """Reinforcement learning agent for maze solving with Q-learning technique.

        Args:
            env_path (str): path to the environment (maze)
            actions (list): actions that can agent make
            display (bool, optional): display initial state of environment. Defaults to True.
        """
        self.environment = Maze(env_path)
        self.actions = actions
        # Initialize Q-table as dictionary (each action is a key)
        self.q_table = {action: np.zeros(self.environment.shape) for action in actions}

        if display:
            self.environment.show_maze("Environment entry")

    
    def learn(self, alpha: float, gamma: float, epsilon_min: float, epochs: int, max_steps: int, first_goal: bool =False, display: list =[]):
        """Let the agent explore the environment and learn how to move in it.

        Args:
            alpha (float): learning rate (changes size of steps in Q-table values adjusting)
            gamma (float): discount factor (determines importance of future step)
            epsilon_min (float): minimum epsilon value (determines probability of exploring)
            epochs (int): number of times agent will go through environment
            max_steps (int): maximum number of steps in an episode
            first_goal (bool, optional): stop learning after hitting the goal for the first time. Defaults to False.
            display (list, optional): display score and agent path in specified epoch. Defaults to [].

        Raises:
            Exception: Reward setting error
        """
        # Set initial epsilon (1 = 100% for exploring)
        self.epsilon = 1

        for epoch in range(1, epochs):
            # Reset environment to initial state
            self.environment.reset()
            score = 0
            steps_count = 0

            while steps_count < max_steps:
                # Current position based on iteration
                current_position = self.environment.current_pos
                # Determine next step
                next_step = self._choose_action(current_position)
                # Make the move
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
                
                ### ADJUSTING Q-TABLE
                # Q-value for current position, next step combination
                current_Q = self.q_table[next_step][current_position]
                # Finding best following step
                next_Q = -float("inf")
                for action in self.actions:
                    if self.q_table[action][next_position] > next_Q:
                        next_Q = self.q_table[action][next_position]
                # Update Q-table
                self.q_table[next_step][current_position] = current_Q + alpha * (reward + gamma * next_Q - current_Q)
                
                score += reward
                steps_count += 1

                 # Goal has been reached
                if finish:
                    break

            # Display info about epoch
            if epoch in display:
                print(f"Epoch: {epoch}, Score: {score}")
                self.environment.show_path(f"Agent path at epoch {epoch}")

            # Stop learning when hitting goal
            if first_goal and finish:
                print(f"Goal was reached at epoch {epoch}")
                break

            # Decrease epsilon (exploring probability)
            if self.epsilon > epsilon_min:
                self.epsilon -= (1 - epsilon_min) / epochs


    def _choose_action(self, position: tuple) -> str:
        """Chooses agents next action.

        Args:
            position (tuple): agents current position

        Returns:
            str: next step (up / down / left / right)
        """
        # Exploration (random move)
        if np.random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        # Exploitation (move is chosen based on Q-table records)
        else:
            # Find best move
            best_Q = -float("inf")
            best_action = None
            for action in self.actions:
                if self.q_table[action][position] > best_Q:
                    best_Q = self.q_table[action][position]
                    best_action = action
            return best_action
        

    def display_path(self):
        """Displays agents path trough the environment (maze).
        """
        print(f"Moves taken: {self.environment.moves_history}")
        print(f"Number of moves: {len(self.environment.moves_history)}")
        self.environment.show_path()