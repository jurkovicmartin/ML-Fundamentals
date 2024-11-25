import numpy as np
import random


class Chromosome:
    def __init__(self, size: tuple, start: tuple =None, finish: tuple =None):
        rows, columns = size
        self.genes = np.array([np.random.choice([0, 1], size=columns) for _ in range(rows)])


    def mutate(self, probability: float):
        flatten_genes = self.genes.flatten()

        for i in range(flatten_genes.size):
            if random.uniform(0, 1) <= probability:
                # Invert the value
                flatten_genes[i] = 1 - flatten_genes[i]

        self.genes = flatten_genes.reshape(self.genes.shape)


    def cross(self, second_chromosome):
        rows, _ = self.genes.shape
        # 0 keeps, 1 takes
        mask = np.random.choice([0, 1], size=rows)

        for i, row in enumerate(mask):
            if row == 1:
                self.genes[i] = second_chromosome.genes[i]
