from population import Population


def main():
    size = (20, 20)
    start = (2, 3)
    finish = (18, 17)
    population = Population(size=30, maze_shape=size, start_point=start, finish_point=finish)
    population.evolve(10, 0.7, 20)
    
    maze, path = population.get_best_maze(True, True)
    print(f"Number of moves to finish {len(path)}")


if __name__ == "__main__":
    main()