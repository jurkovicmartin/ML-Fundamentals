import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from a_star import Astar
from maze import Maze

class Chromosome:
    def __init__(self, size: tuple, start: tuple =None, finish: tuple =None):
        rows, columns = size
        # -2 rows and columns to make space for border
        self.genes = np.array([np.random.choice([0, 2], size=columns - 2) for _ in range(rows - 2)])
        self.genes = np.pad(self.genes, pad_width=1, mode="constant", constant_values=0)
        # Start
        self.genes[start] = 1
        self.genes[finish] = 3

        self.fitness = 0

    
    def set_fitness(self):
        self.fitness = self.get_fitness()
        

    def mutate(self, probability: float):
        # Get rid of border which cannot mutate
        self.genes= self.genes[1: -1, 1: -1]
        flatten_genes = self.genes.flatten()

        for i in range(flatten_genes.size):
            if random.uniform(0, 1) <= probability:
                # Skip start and finish point
                if flatten_genes[i] == 1 or flatten_genes[i] == 3:
                    continue
                # Invert the value
                flatten_genes[i] = 2 - flatten_genes[i]
        # Reshape back and get back border
        self.genes = flatten_genes.reshape(self.genes.shape)
        self.genes = np.pad(self.genes, pad_width=1, mode="constant", constant_values=0)


    def crossover(self, second_chromosome):
        rows, _ = self.genes.shape
        # 0 keeps, 1 takes
        mask = np.random.choice([0, 1], size=rows)

        for i, row in enumerate(mask):
            if row == 1:
                self.genes[i] = second_chromosome.genes[i]


    def show_maze(self, path: bool =False):
        maze = Maze(self.genes)
        maze.show_maze()
        if path:
            search =  Astar(values=self.genes)
            moves = search.find_path()
            for move in moves:
                maze.move(move)
            maze.show_path()


    def get_fitness(self) -> int:
        search = Astar(values=self.genes)
        path = search.find_path()
        # Solution found
        if len(path) > 1:
            # Number of steps
            return len(path)
        # Solution not found
        elif len(path) == 1 and path[0] != 0:
            return 1 - path[0]
        # Solution not found and the start is isolated (0 available moves)
        else:
            return -self.genes.size



    def __str__(self) -> str:
        # String of values (genes) without []
        return " " + np.array2string(self.genes, separator=' ', formatter={'all': lambda x: str(x)}).replace('[', '').replace(']', '')


    def __eq__(self, chromosome):
        return self.fitness == chromosome.fitness
    
    
    def __lt__(self, chromosome):
        return self.fitness < chromosome.fitness