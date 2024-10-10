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

    perceptron.learn(training_data, labels_or)

    perceptron.train(training_data, True)



if __name__ == "__main__":
    main()