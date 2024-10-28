import numpy as np

import matplotlib.pyplot as plt

class Kohonen:
    def __init__(self, input_size: int, neurons_number: int):
        """Kohonen network (self-organizing map) for data clustering. Map has square shape.

        Args:
            input_size (int): number of input values
            neurons_number (int): number of neurons = number of clusters (must be square of integer)

        Raises:
            Exception: neurons_number isn't square
        """
        # Side of square map
        map_size = np.sqrt(neurons_number)
        if not map_size.is_integer():
            raise Exception("This network works with square layout of neurons. Square root of neurons_number must be an integer.")
        # Dimension (input_size, map_size, map_size)
        self.weights = np.array([np.random.rand(int(map_size), int(map_size)) - 0.5 for _ in range(input_size)])


    def train(self, data, learning_rate: float, decrease: float =None, ending: float =None, surrounding: int =0, epochs: int =None):
        """Train network. Training is controlled either by number of epochs or combination of decrease and ending values.

        Args:
            data (array): training data
            learning_rate (float): influences adjusting weights
            decrease (float, optional): decrease of learning rate after each epoch. Defaults to None.
                                        When training is controlled by epochs set decrease is ignored.
                                        When training is controlled by ending, decrease has to be set.
            ending (float, optional): when learning rate hits this value the training ends. Defaults to None.
            surrounding (int, optional): adjusting weights in the surrounding of the winning neuron. Defaults to 0.
                                         Surrounding decreases after each epoch (till 0 = adjusting only winning neuron).
            epochs (int, optional): number of training epochs. Defaults to None.

        Raises:
            Exception: both ending and epochs are set
            Exception: training is controlled by ending but decrease value has not been set
        """
        self.learning_rate = learning_rate
        self.surrounding = surrounding
        # Training controlled by epochs
        if epochs is not None and ending is None:
            for epoch in range(epochs):
                print(f"epoch: {epoch + 1}")

                for sample in data:
                    sample = sample[:, np.newaxis, np.newaxis]
                    
                    win_row, win_column = self.get_winning_neuron(sample)

                    self._update_weights(win_row, win_column, sample)                   

                # Decreasing learning rate
                self.learning_rate = learning_rate * (1 - epoch / epochs)
                if self.surrounding != 0: self.surrounding -= 1

        # Training controlled by ending
        elif epochs is None and ending is not None:
            if decrease is None:
                raise Exception("When training controlled by ending you have to provide decrease value.")
            
            epochs_count = 1
            while self.learning_rate > ending:
                print(f"epoch: {epochs_count}")

                for sample in data:
                    sample = sample[:, np.newaxis, np.newaxis]

                    win_row, win_column = self.get_winning_neuron(sample)

                    self._update_weights(win_row, win_column, sample)

                self.learning_rate -= decrease
                if self.surrounding != 0: self.surrounding -= 1
                epochs_count += 1

        else:
            raise Exception("Training is controlled by either epochs or ending.")
        

    def get_winning_neuron(self, input) -> tuple:
        """Get indexes of winning neuron (closest one to the input value).

        Args:
            input (array): input values (make sure that input dimension is (input_size, 1, 1))

        Returns:
            tuple: row and column of winning neuron
        """
        # Calculating distances between input and all neurons
        distances = np.square(input - self.weights)
        distances = np.sum(distances, axis=0)
        # Closest neuron is the winner
        minimum_index = np.argmin(distances)
        # Return position indexes
        return np.unravel_index(minimum_index, distances.shape)
    

    def _update_weights(self, row: int, column: int, input):
        """Updates the weights. Takes surrounding into account.

        Args:
            row (int): row of neuron to be updated
            column (int): column of neuron to be updated
            input (array): input value (for that neuron)
        """
        # Number of input values (each has its own weights)
        dim, _, _ = (np.shape(self.weights))

        for i in range(dim):
            # Set surrounding boundaries
            row_start = max(0, row - self.surrounding)
            row_end = min(self.weights[i].shape[0], row + self.surrounding + 1)
            col_start = max(0, column - self.surrounding)
            col_end = min(self.weights[i].shape[1], column + self.surrounding + 1)
            # Update weights
            for r in range(row_start, row_end):
                for c in range(col_start, col_end):
                    weight = self.weights[i, r, c]
                    self.weights[i, r, c] = weight + self.learning_rate * (input[i] - weight)
     
