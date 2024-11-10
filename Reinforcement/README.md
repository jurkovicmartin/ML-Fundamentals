Rename Multilayer folder to backpropagation

best record 44 moves


# Reinforcement learning
Q-learning

Agent: moves
actions: up, down, left, right
environment: maze
Q-table: keeps track of outcomes of each move

Exploration: random move
Exploitation: move based on best Q-table record

Updating Q-table: with learning rate and new outcome with record

Greedy epsilon: epsilon value which determines if agent will explore or exploit (probability), epsilon decrease after each epoch

Set max steps and epochs.

Rewards:
move: -1
wall: -10
finish: +10

?? Q function value is corresponds with move outcome, respects the next step ??

Bellman equation