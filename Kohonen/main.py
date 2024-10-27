import numpy as np
import matplotlib.pyplot as plt

from network import Kohonen

def main():
    x = np.random.uniform(-100.0, 100.0, 500)
    y = np.random.uniform(-100.0, 100.0, 500)

    data = np.column_stack((x, y))

    # plt.scatter(x, y)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.show()

    network = Kohonen(2, 25)
    
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, color="blue")
    plt.scatter(network.weights[0], network.weights[1], color="red")

    network.train(data, 0.01, epochs=200)
    plt.subplot(1, 2, 2)
    plt.scatter(x, y, color="blue")
    plt.scatter(network.weights[0], network.weights[1], color="red")

    plt.show()

    input = np.array([[69],[69]])
    row, col = network.get_position(input)
    plt.scatter(input[0], input[1], color="blue")
    plt.scatter(network.weights[0], network.weights[1], color="red")
    plt.scatter(network.weights[0][row][col], network.weights[1][row][col], color="green")

    input = np.array([[-16],[2]])
    row, col = network.get_position(input)
    plt.scatter(input[0], input[1], color="blue")
    plt.scatter(network.weights[0], network.weights[1], color="red")
    plt.scatter(network.weights[0][row][col], network.weights[1][row][col], color="green")


    plt.show()


if __name__ == "__main__":
    main();