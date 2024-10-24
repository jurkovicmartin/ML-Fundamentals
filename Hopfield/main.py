import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

from network import Hopfield

def load_images(folder_path: str) -> list:
    """Loads all png images from a folder.

    Args:
        folder_path (str): path of the folder

    Returns:
        list of loaded images
    """
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            images.append(np.array(Image.open(img_path).convert("L")))
    return images


def main():
    images = load_images("Hopfield/data")
    # Normalize
    images = [(img / 255).flatten() for img in images]
    # Replace 0 with -1 (because of bipolar network function)
    images = [np.where(img == 1, img, -1) for img in images]
    images = np.array(images)

    _, network_size = np.shape(images)

    network = Hopfield(network_size)
    network.train(images)

    ### TESTING

    test_images = load_images("Hopfield/test")
    # Normalize
    test_images = [(img / 255).flatten() for img in test_images]
    # Replace 0 with -1 (because of bipolar network function)
    test_images = [np.where(img == 1, img, -1) for img in test_images]
    test_images = np.array(test_images)
    
    # Network size is image_size * image_size
    image_size = (int(np.sqrt(network_size)), int(np.sqrt(network_size)))

    for image in test_images:
        output = network.reconstruct(image)

        plt.subplot(1, 2, 1)
        plt.imshow(np.reshape(image, image_size), cmap="gray")
        plt.axis("off")
        plt.title("Input image")

        plt.subplot(1, 2, 2)
        plt.imshow(np.reshape(output, image_size), cmap="gray")
        plt.axis("off")
        plt.title("Output image")

        plt.tight_layout()
        plt.show()


    ### RECONSTRUCTION WITH DISPLAYING ALL STATES
    outputs = network.reconstruct(test_images[3], "all")
    for i, out in enumerate(outputs):
        plt.imshow(np.reshape(out, image_size), cmap="gray")
        plt.axis("off")
        plt.title(f"Image at {i} iteration")
        plt.show()




if __name__ == "__main__":
    main()