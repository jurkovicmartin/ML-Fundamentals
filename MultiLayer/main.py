import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from network import NeuralNetwork


def main():
    """
    Basic model usage on 18x18 images from folder data.
    """
    images = []
    for i in range(10):
        # Load 18x18 image and covert it to grayscale
        image = Image.open(f"MultiLayer/data/{i}.bmp").convert("L")
        image = np.array(image)
        # Normalize
        image = image
        # Flatten the matrix intro vector
        image = image.flatten()

        images.append(image)

    # 10x324
    images = np.array(images)

    # Matrix with 1 one main diagonal
    labels = np.eye(10)

    neurons_num = [40, 20]

    network = NeuralNetwork(input_num=324, output_num=10, hidden_num=2, hidden_neurons=neurons_num)
    network.train(data=images, labels=labels, learning_rate=0.1, momentum=0.2, acceptable_error=0.1, graph=True)

    ### TESTING
    # Simple testing without using test method from the model
    noisy_images = []
    for image in images:
        noise = np.random.uniform(-0.7, 0.7, image.shape)
        # Make sure the values are in range 0-1
        noisy_images.append(np.clip(image + noise, 0, 1))

    noisy_images = np.array(noisy_images)

    for image in noisy_images:
        prediction = network.predict(image[:, np.newaxis], "output")
        predicted_value = np.argmax(prediction)

        plt.imshow(image.reshape(18,18), cmap="gray")
        plt.axis("off")
        plt.title("Input image")
        plt.show()

        print(f"Predicted value is {predicted_value} with probability {prediction[predicted_value]}")
        
        plt.bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], prediction.flatten())
        plt.title("Output predictions")
        plt.xlabel("Value")
        plt.ylabel("Probability")
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        plt.show()


if __name__ == "__main__":
    main()