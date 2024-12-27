import random

from chromosome import Chromosome
from a_star import Astar


class Population:
    def __init__(self, size: int, maze_shape: tuple, start_point: tuple =None, finish_point: tuple =None):
        """Population of chromosomes to generate a maze.

        Args:
            size (int): size of the population
            maze_shape (tuple): size of the maze
            start_point (tuple, optional): start point. Defaults to None = random.
            finish_point (tuple, optional): finish point. Defaults to None = random.
        """
        self.population = [Chromosome(maze_shape, start_point, finish_point) for _ in range(size)]

        for ch in self.population:
            ch.set_fitness()


    def evolve(self, mutation_count: int, mutation_probability: float, crossover_count: int, mutation_extend: bool =True, crossover_extend: bool =True, max_epochs: int =None, mutation_weight: float =0.5, crossover_weight: float =0.35):
        """Evolve the population.

        Args:
            mutation_count (int): number of chromosomes to be mutated
            mutation_probability (float): probability of chromosome gene mutation
            crossover_count (int): number of chromosomes to be crossed over
            mutation_extend (bool, optional): if mutating should keep the original (increases population size). Defaults to True.
            crossover_extend (bool, optional): if crossing over should keep the original (increases population size). Defaults to True.
            max_epochs (int, optional): maximum number of epochs. Defaults to None = until solution is found.
            mutation_weight (float, optional): weight determines mutation probability for chromosomes with solution (weight * probability)
            crossover_weight (float, optional): determines probability of chromosome keeping his genes (0 + weight, 1 - weight)

        """
        self.mutation_count = mutation_count
        self.mutation_probability = mutation_probability
        self.crossover_count = crossover_count
        self.mutation_extend = mutation_extend
        self.crossover_extend = crossover_extend
        self.mutation_weight = mutation_weight
        self.crossover_weight = crossover_weight

        epochs = 1
        # Go until there is a chromosome with solution
        while max(self.population).fitness < 0:

            if max_epochs and max_epochs == epochs:
                return
            
            print(f"Epoch: {epochs}")
            self._mutate()

            self._crossover()

            self._tournament()

            epochs += 1


    def _mutate(self):
        """Mutates chromosomes in the population.
        """
        # Indexes of chromosomes to be mutated
        indexes = self._random_indexes(self.population, self.mutation_count)

        # Extending population
        if self.mutation_extend:
            for idx in indexes:
                self.population.append(self.population[idx])
                self.population[-1].mutate(self.mutation_probability, self.mutation_weight)
                self.population[-1].set_fitness()
        else:
            for idx in indexes:
                self.population[idx].mutate(self.mutation_probability, self.mutation_weight)
                self.population[idx].set_fitness()


    def _crossover(self):
        """Performs a cross over in the population.

        Raises:
            Exception: Crossover too high for population size
        """
        if self.crossover_count > len(self.population) / 2:
            raise Exception("Cannot cross this many. Maximum is half of the population")
        
        # Take first half of chromosomes to cross
        indexes = self._random_indexes(self.population, self.crossover_count)
        indexes.sort(reverse=True)

        population_copy = self.population.copy()
        taken_chromosomes = [population_copy.pop(idx) for idx in indexes]

        # Population hasn't been split in half
        if len(population_copy) != len(taken_chromosomes):
            # Select random chromosomes from remaining ones
            indexes = self._random_indexes(population_copy, len(taken_chromosomes))
            indexes.sort(reverse=True)
            chosen_chromosomes = [population_copy.pop(idx) for idx in indexes]
        # Population has been split in half
        else:
            chosen_chromosomes = population_copy

        # Extending population
        if self.crossover_extend:
            for i in range(self.crossover_count):
                self.population.append(taken_chromosomes[i])
                self.population[-1].crossover(chosen_chromosomes[i], self.crossover_weight)
                self.population[-1].set_fitness()
        else:
            crosses = 0
            for ch in self.population:
                if crosses == self.crossover_count:
                    break

                if ch in taken_chromosomes:
                    ch.crossover(chosen_chromosomes[crosses], self.crossover_weight)
                    ch.set_fitness()
                    crosses += 1
        
    
    def _tournament(self):
        """Performs a tournament selection on the population.
        """
        random.shuffle(self.population)

        first_half = self.population[:len(self.population) // 2]
        second_half = self.population[len(self.population) // 2:]
        # Comparing with fitness
        self.population = [ch1 if ch1 > ch2 else ch2 for ch1, ch2 in zip(first_half, second_half)]
    

    def get_best_maze(self, show: bool =True, show_path: bool =False) -> tuple:
        """Get best chromosome (maze) in the population.

        Args:
            show (bool, optional): Show the best maze. Defaults to True.
            show_path (bool, optional): Show also the solving path (). Defaults to False.

        Returns:
            tuple: (maze matrix, list of moves)
        """
        best_chromosome = max(self.population)

        search = Astar(values=best_chromosome.genes)
        path = search.find_path()

        if show and not show_path:
            best_chromosome.show_maze()

        if show and show_path:
            best_chromosome.show_maze(show_path)

        return best_chromosome.genes, path


    @staticmethod
    def _random_indexes(list: list, count: int) -> list:
        """Select random indexes from a list.

        Args:
            list (list): list from which take the indexes
            count (int): number of indexes to be selected

        Returns:
            list: list of selected indexes
        """
        return random.sample(range(len(list)), count)
