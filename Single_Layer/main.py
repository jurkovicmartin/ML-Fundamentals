import numpy as np
import matplotlib.pyplot as plt

from single_layer import SingleLayerNetwork


def main():
    training_data = [[1, 1, 1, 1, 1], # 0
                     [0, 1, 1, 1, 1], # 1
                     [0, 0, 1, 1, 1], # 2
                     [0, 0, 0, 1, 1], # 3
                     [0, 0, 0, 0, 1], # 4
                     [0, 0, 0, 0, 0], # 5
                     [1, 0, 0, 0, 0], # 6
                     [1, 1, 0, 0, 0], # 7
                     [1, 1, 1, 0, 0], # 8
                     [1, 1, 1, 1, 0]] # 9
    # Matrix with 1 on the main diagonal
    labels = np.eye(10)

    network = SingleLayerNetwork(neurons=10, input_size=5)
    network.train(data=training_data, labels=labels, learning_rate=1, error=0.1, graph=True)


    ### TESTING

    testing_data = [[0, 0, 0, 1, 1], # 3
                    [1, 1, 1, 1, 1], # 0
                    [0, 0, 1, 1, 1], # 2
                    [1, 1, 1, 1, 0]] # 9
    # Adding noise
    testing_data = [[x + np.random.uniform(-0.3, 0.3) for x in row] for row in testing_data]
    testing_labels = [3, 0, 2, 9]
    error =  0
    
    for idx, sample in enumerate(testing_data):
        prediction = network.predict(sample).flatten()

        # Index in prediction corresponds to the value
        predicted_value = np.argmax(prediction)

        if predicted_value != testing_labels[idx]:
            error += 1

        print(f"Predicted value for sample {idx} is {predicted_value} with probability {prediction[predicted_value]}")

        plt.bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], prediction)
        plt.title(f"Sample {idx} prediction")
        plt.ylabel("Probability")
        plt.xlabel("Value")
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        plt.show()

    samples_num = len(testing_data)
    accuracy = (samples_num - error) / samples_num
    print(f"Network accuracy is {accuracy} with {error} errors from {samples_num} samples.")



if __name__ == "__main__":
    main()