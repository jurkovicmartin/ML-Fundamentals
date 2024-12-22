from population import Population


def main():
    size = (20, 20)
    start = (2, 3)
    finish = (17, 18)
    population = Population(size=30, maze_shape=size, start_point=start, finish_point=finish)
    population.evolve(mutation_count=10, mutation_probability=0.7, crossover_count=10)
    
    maze, path = population.get_best_maze(True, True)
    print(f"Number of moves to finish {len(path)}")


if __name__ == "__main__":
    main()