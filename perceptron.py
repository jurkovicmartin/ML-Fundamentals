import numpy as np
import matplotlib.pyplot as plt

from functions import step_function

class Perceptron:
    def __init__(self, input_size: int, weights =None, bias: float =None,  learning_rate: float =0.1):
        """
        Parameters
        ------
        input_size: number of input values

        weights: array, starting weights, None => random weights +(0 - 1), !weights must a vector with size equals to input_size

        bias: starting bias, None => random bias +(0 - 1)

        learning_rate: effects adjusting of weights + bias while learning
        """
        if weights is None:
            self.weights = np.random.rand(input_size)
        elif np.array(weights).shape == (input_size,):
            self.weights = np.array(weights)
        else:
            raise Exception("Weights must be a vector with size equals to input_size")
        
        if bias is None:
            self.bias = np.random.rand(1)
        else:
            self.bias = bias

        self.learning_rate = learning_rate

        self.ACTIVATION_FUNCTION = step_function


    def predict(self, input):
        # z = wi * xi + w0
        z = np.dot(self.weights, input) + self.bias

        return self.ACTIVATION_FUNCTION(z)
    

    def learn(self, data, labels, epochs: int =None):
        """
        Parameters
        -----
        data: learning data

        labels: expected output for training data

        epochs: number of times a model sees the learning sequence, None => till weights and bias is changing
        """
        print("Learning")
        if epochs is None:
            temp_weights = -1
            temp_bias = -1
            epochs_count = 1
            while not np.array_equal(temp_weights, self.weights) or  not np.array_equal(temp_bias, self.bias):
                print(f"epoch: {epochs_count}")

                temp_weights = np.copy(self.weights)
                temp_bias = np.copy(self.bias)

                for input, label in zip(data, labels):

                    prediction = self.predict(input)

                    error = prediction - label

                    self.weights -= error * self.learning_rate * np.array(input)
                    self.bias -= error * self.learning_rate

                epochs_count += 1
                
        else:
            for epoch in range(epochs):
                print(f"epoch: {epoch + 1}")
                for input, label in zip(data, labels):

                    prediction = self.predict(input)

                    error = prediction - label

                    self.weights -= error * self.learning_rate * np.array(input)
                    self.bias -= error * self.learning_rate


    def train(self, data, graph: bool =False):
        """
        data: training data

        graph: show graphical interpretation
        """
        print("Training")
        if graph:
            x1 = []
            x2 = []
            y = []
            for input in data:
                prediction = self.predict(input)
                print(f"Input: {input}; Prediction: {prediction}")

                x1.append(input[0])
                x2.append(input[1])
                y.append(prediction)
            
            plt.scatter(np.array(x1), np.array(x2), c=np.array(y), cmap="coolwarm_r")
            plt.title("Training output")
            plt.xlim(-1, 2)
            plt.ylim(-1, 2)
            plt.colorbar(label='y value')  # Show color scale for y
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.show()

        else:
            for input in data:
                prediction = self.predict(input)
                print(f"Input: {input}; Prediction: {prediction}")