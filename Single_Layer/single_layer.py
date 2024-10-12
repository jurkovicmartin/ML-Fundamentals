import numpy as np
import matplotlib.pyplot as plt


def activation_function(z: float) -> float:
    """
    Sigmoid function.
    y(x) = 1 / (1 + e^-x)
    """
    return 1 / (1 + np.exp(-z))


def derivation_function(z: float) -> float:
    """
    Derivation of activation function.
    Derivation of sigmoid.
    y(x) = 1 / (1 + e^-x)
    y(x)' = y(x) * (1 - y(x))
    """
    return activation_function(z) * (1 - activation_function(z))


class SingleLayerNetwork:
    def __init__(self, neurons: int, input_size: int, weights =None, biases =None):
        """Represents a single layer neural network

        Args:
            neurons (int): number of neurons in the layer
            input_size (int): number of inputs values
            weights (matrix, optional): starting weights of the neurons. Defaults to None => random weights.
            biases (array, optional): starting biases of neurons. Defaults to None => random biases.
        """
        if weights is None:
            self.weights = np.random.rand(neurons, input_size)
        elif np.array(weights).shape == (neurons, input_size):
            self.weights = np.array(weights)
        else:
            raise Exception("Wrong weights matrix shape. Weights must have shape (neurons, input_size)")
        
        if biases is None:
            # Makes self.bias shape (n, 1) instead of (n,)
            self.biases = np.random.rand(neurons)[:, np.newaxis]
        elif np.array(biases).shape == (neurons, 1):
            self.biases = np.array(biases)
        else:
            raise Exception("Wrong biases array shape. Biases must have shape (neurons, 1)")
        

    def predict(self, input):
        """
        Returns array of predictions.
        """
        input = np.array(input)[:, np.newaxis]
        # z = wi * xi + b
        z = np.dot(self.weights, input) + self.biases

        return np.array([activation_function(x) for x in z])
    

    def train(self, data, labels, learning_rate: float =0.1, epochs: int =None, error: float =None, graph: bool =None):
        """Train the network. You need to provide number of epochs or error value. One of thee will determine when to stop training.

        Args:
            data (matrix): training samples
            labels (array): expected outputs (for training samples)
            learning_rate (float, optional): effects adjusting weights and biases. Defaults to 0.1.
            epochs (int, optional): number of times network sees the training samples. Defaults to None.
            error (float, optional): value of acceptable network error value (trains till reach this error). Defaults to None.
            graph (bool, optional): show graph of network error developing over epochs
        """
        print("Training")
        # Training controlled by number of epochs
        if epochs is not None and error is None:
            for epoch in range(epochs):
                network_error = 0
                print(f"epoch {epoch + 1}")
                for input, label in zip(data, labels):
                    input = np.array(input)
                    label = np.array(label)[:, np.newaxis]

                    outputs = self.predict(input)
                    
                    neurons_errors = np.array(label - outputs)

                    sample_error = 0.5 * np.sum(np.power(neurons_errors, 2))
                
                    # derivations = np.array([derivation_function(x) for x in neurons_errors])
                    # self.weights += learning_rate * derivations * input
                    # self.biases += learning_rate * derivations

                    derivations = np.array([derivation_function(x) for x in outputs])
                    # Makes input shape (1, n) instead of (n,) 
                    self.weights += learning_rate * neurons_errors * derivations * input.reshape(1, -1)
                    self.biases += learning_rate * neurons_errors * derivations

                    network_error += sample_error

                print(f"Network error {network_error}")

                if graph:
                    plt.scatter(epoch + 1, network_error, color="blue", s=10)
                    plt.title("Network error development.")
                    plt.xlabel("Epochs")
                    plt.ylabel("Error")

        # Training controlled by network error value
        elif epochs is None and error is not None:
            # Set very big to pass the first while condition (for sure)
            network_error = float("inf")
            epochs_count = 1
            while network_error > error:
                # Reset the value for each epoch
                network_error = 0
                print(f"epoch: {epochs_count}")

                for input, label in zip(data, labels):
                    input = np.array(input)
                    label = np.array(label)[:, np.newaxis]

                    outputs = self.predict(input)
                    
                    neurons_errors = np.array(label - outputs)

                    sample_error = 0.5 * np.sum(np.power(neurons_errors, 2))

                    derivations = np.array([derivation_function(x) for x in outputs])
                    # Makes input shape (1, n) instead of (n,) 
                    self.weights += learning_rate * neurons_errors * derivations * input.reshape(1, -1)
                    self.biases += learning_rate * neurons_errors * derivations

                    network_error += sample_error

                epochs_count += 1
                print(f"Network error {network_error}")

                if graph:
                    plt.scatter(epochs_count, network_error, color="blue", s=10)
                    plt.title("Network error development.")
                    plt.xlabel("Epochs")
                    plt.ylabel("Error")

        else:
            raise Exception("Training error. You need to provide number of epochs or error value.")
        
        if graph:
            plt.show()