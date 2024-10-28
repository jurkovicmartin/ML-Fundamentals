import numpy as np
import matplotlib.pyplot as plt

from network import Kohonen

def main():
    # Random data generation
    x = np.random.uniform(-100.0, 100.0, 500)
    y = np.random.uniform(-100.0, 100.0, 500)
    data = np.column_stack((x, y))


    network = Kohonen(2, 25)
    # Show data with initial neurons positions
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, color="blue", label="data", s=10)
    plt.scatter(network.weights[0], network.weights[1], color="red", label="neurons", s=15)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Neurons layout at the start")
    plt.legend(loc="upper left")

    ### TRAINING

    network.train(data=data, learning_rate=1, decrease=0.01, ending=0.1, surrounding=1, epochs=None)
    # Show data with trained neurons
    plt.subplot(1, 2, 2)
    plt.scatter(x, y, color="blue", label="data", s=10)
    plt.scatter(network.weights[0], network.weights[1], color="red", label="neurons", s=15)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Neurons layout at the end")
    plt.legend(loc="upper left")
    plt.show()

    ### TESTING

    input = np.array([[13],[-26]])
    row, col = network.get_winning_neuron(input[:, np.newaxis])
    # Show input value and neurons
    plt.scatter(input[0], input[1], color="blue", label="input")
    plt.scatter(network.weights[0], network.weights[1], color="red", label="neurons")
    plt.scatter(network.weights[0][row][col], network.weights[1][row][col], color="green", label="winning neuron")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Illustration of network output")
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main();