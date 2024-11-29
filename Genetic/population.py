import random

from chromosome import Chromosome
from a_star import Astar


class Population:
    def __init__(self, size: int, maze_shape: tuple, start_point: tuple =None, finish_point: tuple =None):
        self.population = [Chromosome(maze_shape, start_point, finish_point) for _ in range(size)]

        for ch in self.population:
            ch.set_fitness()


    def evolve(self, mutation_count: int, mutation_probability: float, crossover_count: int, mutation_extend: bool =True, crossover_extend: bool =True):
        epochs = 1
        while max(self.population).fitness < 0:
            print(f"Epoch: {epochs}")
            self._mutate(mutation_count, mutation_probability, mutation_extend)

            self._crossover(crossover_count, crossover_extend)

            self._tournament()

            epochs += 1


    def _mutate(self, count: int, probability: float, extend: bool =True):
        indexes = self._random_indexes(self.population, count)

        # Extending population
        if extend:
            for idx in indexes:
                self.population.append(self.population[idx])
                self.population[-1].mutate(probability)
                self.population[-1].set_fitness()
        else:
            for idx in indexes:
                self.population[idx].mutate(probability)
                self.population[idx].set_fitness()


    def _crossover(self, count: int, extend: bool =True):
        if count > len(self.population) / 2:
            raise Exception("Cannot cross this many. Maximum is half of the population")
        
        indexes = self._random_indexes(self.population, count)
        indexes.sort(reverse=True)

        population_copy = self.population.copy()
        taken_chromosomes = [population_copy.pop(idx) for idx in indexes]

        if len(population_copy) != len(taken_chromosomes):
            indexes = self._random_indexes(population_copy, len(taken_chromosomes))
            indexes.sort(reverse=True)
            chosen_chromosomes = [population_copy.pop(idx) for idx in indexes]
        else:
            chosen_chromosomes = population_copy

        if extend:
            for i in range(count):
                self.population.append(taken_chromosomes[i])
                self.population[-1].crossover(chosen_chromosomes[i])
                self.population[-1].set_fitness()
        else:
            crosses = 0
            for ch in self.population:
                if crosses == count:
                    break

                if ch in taken_chromosomes:
                    ch.crossover(chosen_chromosomes[crosses])
                    ch.set_fitness()
                    crosses += 1
        
    
    def _tournament(self):
        random.shuffle(self.population)

        first_half = self.population[:len(self.population) // 2]
        second_half = self.population[len(self.population) // 2:]

        self.population = [ch1 if ch1 > ch2 else ch2 for ch1, ch2 in zip(first_half, second_half)]
    

    def get_best_maze(self, show: bool =True, show_path: bool =False) -> tuple:
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
        indexes = []
        while len(indexes) < count:
            idx = random.randint(0, len(list) - 1)
            if idx not in indexes:
                indexes.append(idx)
        return indexes
