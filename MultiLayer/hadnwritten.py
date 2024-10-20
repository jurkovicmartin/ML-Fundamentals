import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from network import NeuralNetwork


def main():
    """
    Handwritten digits recognition with Mnist dataset.
    """
    # Mnist dataset of 28x28 images with handwritten digits and labels
    mnist = tf.keras.datasets.mnist
    # X: pixel image data, Y: digit labels
    # 60 000 of training data nad 10 000 of testing data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    ### INPUT PREPROCESSING

    # Normalizing (from 0-255 to 0-1)
    x_train = tf.keras.utils.normalize(x_train, axis=1)

    # Reshaping from (60000, 28, 28) to (60000, 784) = flattening the images
    x_train = x_train.reshape(len(x_train), 784)

    # Each row represents one digit (corresponds to one output neuron activation)
    digits_labels = np.eye(10)
    # y_train contains only values => replacing these values with matrixes resulting in (60000, 10)
    y_train_mat = np.array([digits_labels[x] for x in y_train])

    # Slicing the training data to 1000 samples
    input_slice = x_train[0:1000]
    labels_slice = y_train_mat[0:1000]

    ### NEURAL NETWORK

    network = NeuralNetwork(input_num=784, output_num=10, hidden_num=2, hidden_neurons=[40, 20])
    # Training
    network.train(data=input_slice, labels=labels_slice, learning_rate=0.01, momentum=0.5, acceptable_error=0.1, graph=True)

    ### TESTING

    # Taking testing data from training dataset (because it is already flatten and whole dataset isn't used for training)
    test_data_slice = x_train[2000:2100]
    # Labels taking from original y_train (because we need only the value not activation of each neuron)
    test_labels_slice = y_train[2000:2100]

    accuracy, samples, errors_count, errors = network.test(test_data_slice, test_labels_slice, "detailed")
    print(f"Accuracy: {accuracy}\nNumber of samples: {samples}\nNumber of errors: {errors_count}")

    # Plot frequency of errors
    plt.bar(errors.keys(), errors.values())
    plt.title("Frequency of the errors")
    plt.xlabel("Value")
    plt.ylabel("Errors count")
    plt.xticks(list(errors.keys()))
    plt.grid(True, axis="y")
    plt.show()


if __name__ == "__main__":
    main()
