import numpy as np
import random

from a_star import Astar
from maze import Maze

class Chromosome:
    def __init__(self, size: tuple, start: tuple =None, finish: tuple =None):
        """Chromosome representing a maze.

        Args:
            size (tuple): size of the maze (rows x columns)
            start (tuple, optional): starting position. Defaults to None = random.
            finish (tuple, optional): finish position. Defaults to None = random.
        """
        rows, columns = size
        # -2 rows and columns to make space for border
        self.genes = np.array([np.random.choice([0, 2], size=columns - 2) for _ in range(rows - 2)])
        self.genes = np.pad(self.genes, pad_width=1, mode="constant", constant_values=0)

        # Random start and finish position
        if not start:
            start = (np.random.randint(1, rows-1), np.random.randint(1, columns-1))
        if not finish:
            finish = (np.random.randint(1, rows-1), np.random.randint(1, columns-1))
            # Start and finish is the same
            while finish == start:
                finish = (np.random.randint(1, rows-1), np.random.randint(1, columns-1))
        # Setting starting and finish position into the maze
        self.genes[start] = 1
        self.genes[finish] = 3

        self.fitness = 0

    
    def set_fitness(self):
        """Sets fitness to the chromosome.
        """
        self.fitness = self.get_fitness()
        

    def mutate(self, probability: float):
        """Mutates genes of the chromosome. The probability is half if the maze has a solution.

        Args:
            probability (float): probability for mutation
        """
        # Get rid of border which cannot mutate
        self.genes= self.genes[1: -1, 1: -1]
        flatten_genes = self.genes.flatten()

        # Lower mutation probability for mazes with solution
        if self.fitness > 0:
            probability = 0.5 * probability

        for i in range(flatten_genes.size):
            if random.uniform(0, 1) <= probability:
                # Skip start and finish point
                if flatten_genes[i] == 1 or flatten_genes[i] == 3:
                    continue
                # Invert the value (mutation)
                flatten_genes[i] = 2 - flatten_genes[i]
        # Reshape back and get back border
        self.genes = flatten_genes.reshape(self.genes.shape)
        self.genes = np.pad(self.genes, pad_width=1, mode="constant", constant_values=0)


    def crossover(self, second_chromosome):
        """Crossing genes of two chromosomes. Changing the original.
        Chromosome with higher fitness has higher probability for its genes to be taken.

        Args:
            second_chromosome (Chromosome): Second chromosome for crossing (this one doesn't change)
        """
        rows, _ = self.genes.shape

        # Creating a crossing mask (0 keeps, 1 takes genes)
        if self.fitness > second_chromosome.fitness:
            mask = np.random.choice([0, 1], size=rows, p=[0.65, 0.35])
        else:
            mask = np.random.choice([0, 1], size=rows, p=[0.35, 0.65])

        for i, row in enumerate(mask):
            if row == 1:
                self.genes[i] = second_chromosome.genes[i]


    def show_maze(self, path: bool =False):
        """Show genes as a maze.

        Args:
            path (bool, optional): Show also found path if exists. Defaults to False.
        """
        maze = Maze(self.genes)
        maze.show_maze()
        if path:
            search =  Astar(values=self.genes)
            moves = search.find_path()

            if len(moves) == 1:
                print("Path doesn't exist")
                return
            
            for move in moves:
                maze.move(move)
            maze.show_path()


    def get_fitness(self) -> int:
        """Get fitness value of the chromosome. Value is based on result of path finding algorithm.

        Returns:
            int: number of steps if maze has solution. Otherwise negative value.
        """
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