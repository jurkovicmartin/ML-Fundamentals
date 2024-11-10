from reinforcement import Agent

def main():
    path = "data/maze1.png"

    agent = Agent(path, ["up", "down", "left", "right"])
    agent.environment.show_maze("Maze entry")
    agent.learn(epsilon=1, alpha=0.7, gamma=0.8, epsilon_min=0.1, episodes=3000, max_steps=100)
    agent.environment.show_maze()
    agent.environment.show_path()
    print(f"Moves to the end: {agent.environment.moves_history}")
    print(f"Number of moves: {len(agent.environment.moves_history)}")



if __name__ == "__main__":
    main()