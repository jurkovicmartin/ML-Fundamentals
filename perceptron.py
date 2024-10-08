import numpy as np

def step_function(x) -> int:
    """
    Activation function
    """
    return 0 if x < 0 else 1


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


    def predict(self, input):
        # z = wi * xi + w0
        z = np.dot(self.weights, input) + self.bias

        return step_function(z)
    

    def learn(self, data, labels, epochs: int):
        print("Learning")
        for epoch in range(epochs):
            print(f"epoch: {epoch + 1}")
            for input, label in zip(data, labels):

                prediction = self.predict(input)

                error = label - prediction

                self.weights += self.learning_rate * error * np.array(input)
                self.bias += self.learning_rate * error

        
            print(f"Adjusted weights: {self.weights}")
            print(f"Adjusted bias: {self.bias}")   


    def train(self, data):
        print("Training")
        for input in data:
            prediction = self.predict(input)
            print(f"Input: {input}; Prediction: {prediction}")