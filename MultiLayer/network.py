import numpy as np
import matplotlib.pyplot as plt

import copy

def activation_function(z: float) -> float:
    """
    Hyperbolic tangent function.
    y(x) = (1 - e^x) / (1 + e^-x)
    """
    return np.tanh(z)


def derivation_function(y: float) -> float:
    """
    Derivation of activation function.
    Derivation of tanh.
    y(x) = (1 - e^x) / (1 + e^-x)
    y(x)' = 1 - y(x)^2

    Parameters
    ----
    y: value of the tanh function
    """
    return 1 - np.square(y)

class NeuralNetwork:
    def __init__(self, input_num: int, output_num: int, hidden_num: int, hidden_neurons):
        """Multilayer neural network with error back propagation. Initial weights are random and biases are 0.

        Args:
            input_num (int): number of network inputs
            output_num (int): number of network outputs (number of neurons in the output layer)
            hidden_num (int): number of hidden layers
            hidden_neurons (array): each item of the array specifies the number of neurons in corresponding layer 
                                    ([2, 3] means first hidden layer has 2 neurons and second has 3)
        """
        if len(hidden_neurons) != hidden_num:
            raise Exception("Length of hidden_neurons must match number of hidden layers.")
        # +1 for output layer
        self.layers = hidden_num + 1

        ### GENERATE WEIGHTS
        # List of weights [layer][neuron][weight]
        self.weights = []
        # First layer
        self.weights.append(0.01 * np.random.rand(hidden_neurons[0], input_num))
        # Other hidden layers
        for i in range(1, hidden_num):
            self.weights.append(0.01 * np.random.rand(hidden_neurons[i], hidden_neurons[i-1]))
        # Output layer
        self.weights.append(0.01 * np.random.rand(output_num, hidden_neurons[-1]))

        ### GENERATE BIASES
        # List of biases [layer][neuron][bias]
        self.biases = [0.01 * np.random.rand(hidden_neurons[i], 1) for i in range(hidden_num)]
        # Add output layer
        self.biases.append(0.01 * np.random.rand(output_num, 1))

    
    def predict(self, input, amount: str):
        """Get predictions from neurons.

        Args:
            input (array): input data
            amount (str): amount of predictions to be returned
                    ('output' for only predictions from output layer / 'all' predictions from all layers)

        Returns:
            array: predictions
        """
        outputs = []
        # First hidden layer
        z = np.dot(self.weights[0], input) + self.biases[0]
        outputs.append(activation_function(z))
        # Other layers
        for i in range(1, self.layers):
            z = np.dot(self.weights[i], outputs[i-1])
            outputs.append(activation_function(z))

        if amount == "output":
            # Return only predictions from output layer
            return outputs[-1]
        elif amount == "all":
            return outputs
        else:
            raise Exception("Invalid amount option.")
        

    def train(self, data, labels, learning_rate: float, acceptable_error: float, momentum: float =0.0, epochs: int =1000, graph: bool =False):
        """Trains the network.

        Args:
            data (array): training data
            labels (array): training labels
            learning_rate (float): effects adjusting steps
            acceptable_error (float): training continues until the network error isn't lower
            momentum (float, optional): effects adjusting steps. Defaults to 0.0.
            epochs (int, optional): maximum number of epochs. Defaults to 1000.
            graph (bool, optional): show network error function. Defaults to False.
        """
        epochs_count = 1
        network_error = float("inf")

        # Just "=" doesn't create copy the list
        last_weights = copy.deepcopy(self.weights)
        last_biases = copy.deepcopy(self.biases)

        while network_error > acceptable_error:
            print(f"epoch {epochs_count}")
            network_error = 0

            for sample, label in zip(data, labels):
                # Reshape
                sample = np.array(sample)[:, np.newaxis]
                label = np.array(label)[:, np.newaxis]

                ### FORWARD PASS
                outputs = self.predict(sample, "all")

                output_error = label - outputs[-1]
                sample_error =np.mean(np.square(output_error))
                network_error += sample_error

                ### ERROR BACK PROPAGATION
                deltas = []
                errors = [output_error]

                # Output layer
                deltas.insert(0, errors[-1] * derivation_function(outputs[-1]))

                # Other layers
                for i in range(1, self.layers):
                    error = np.dot(self.weights[-i].T, errors[0])
                    errors.insert(0, error)
                    delta = error * derivation_function(outputs[-i-1])
                    deltas.insert(0, delta)

                ### ADJUSTING
                # First layer
                
                help_weights = copy.deepcopy(self.weights)
                help_biases = copy.deepcopy(self.biases)

                self.weights[0] += learning_rate * deltas[0] * sample.T + momentum * (self.weights[0] - last_weights[0])
                # print(self.biases[0] == last_biases[0])
                self.biases[0] += learning_rate * deltas[0] + + momentum * (self.biases[0] - last_biases[0])
                # Other layers
                for i in range(1, self.layers):
                    self.weights[i] += learning_rate * deltas[i] * outputs[i-1].T + momentum * (self.weights[i] - last_weights[i])
                    self.biases[i] += learning_rate * deltas[i]+ momentum * (self.biases[i] - last_biases[i])

                last_weights = copy.deepcopy(help_weights)
                last_biases = copy.deepcopy(help_biases)

            if graph:
                plt.scatter(epochs_count, network_error, s=10, color="blue")
                plt.title("Network error function")
                plt.xlabel("Epochs")
                plt.ylabel("Error")

            print(f"network_error {network_error}")

            # Stop training at max epochs
            if epochs_count == epochs:
                break

            epochs_count += 1
        
        if graph:
            plt.show()


        