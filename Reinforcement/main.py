from reinforcement import Agent

def main():
    path = "Reinforcement/data/maze1.png"

    agent = Agent(path, actions=["up", "down", "left", "right"], display=False)
    agent.learn(alpha=0.7, gamma=0.8, epsilon_min=0, epochs=1000, max_steps=100, display=[100, 200, 300, 400, 500, 600, 700, 800, 900])
    print("Final solution:")
    agent.display_path()
    



if __name__ == "__main__":
    main()