import numpy as np


class Kohonen:
    def __init__(self, input_size: int, neurons_number: int):
        map_size = np.sqrt(neurons_number)
        if not map_size.is_integer():
            raise Exception("This network works with square layout of neurons. Square root of neurons_number must be an integer.")
        
        self.weights = np.array([np.random.rand(int(map_size), int(map_size)) - 0.5 for _ in range(input_size)])


    def train(self, data, learning_rate: float, surrounding: int=None, epochs: int=None):
        for epoch in range(epochs):
            print(f"epoch: {epoch + 1}")
            for sample in data:
                sample = sample[:, np.newaxis, np.newaxis]
                distances = np.square(sample - self.weights)
                distances = np.sum(distances, axis=0)

                minimum_index = np.argmin(distances)
                min_row, min_column = np.unravel_index(minimum_index, distances.shape)

                dim, _, _ = (np.shape(self.weights))

                for i in range(dim):
                    weight = self.weights[i][min_row][min_column]
                    weight = weight + learning_rate * (sample[i] - weight)
                    self.weights[i][min_row][min_column] = weight


    def get_position(self, input):
        input = input[:, np.newaxis]
        distances = np.square(input - self.weights)
        distances = np.sum(distances, axis=0)

        minimum_index = np.argmin(distances)
        return np.unravel_index(minimum_index, distances.shape)
     
