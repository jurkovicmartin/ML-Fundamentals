"""
Binary gate approximation with perceptron.
"""

from perceptron import Perceptron
import numpy as np

def main():
    training_data =  np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    # Example of AND gate
    labels_and = np.array([0, 0, 0, 1])
    # Example of OR gate
    labels_or = np.array([0, 1, 1, 1]) 

    perceptron = Perceptron(2)

    perceptron.train(training_data, labels_and)

    perceptron.test(training_data, True)



if __name__ == "__main__":
    main()