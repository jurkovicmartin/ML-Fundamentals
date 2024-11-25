from chromosome import Chromosome


class Population:
    def __init__(self, size: int, maze_shape: tuple):
        self.population = [Chromosome(maze_shape) for _ in range(size)]