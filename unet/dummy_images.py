"""
Module for generating dummy images for sanity checking unet model
"""
import numpy as np
import os
from PIL import Image

def generate_dummy_image(size=(572, 572), num_objects=5, object_size=20):
    """
    Generate a dummy grayscale image with white objects scattered randomly.
    
    Args:
        size (tuple): Size of the generated image (height, width).
        num_objects (int): Number of white objects to scatter in the image.
        object_size (int): Size of each white object (square).
        
    Returns:
        np.ndarray: Generated dummy image.
    """
    image = np.zeros(size, dtype=np.uint8)
    height, width = size
    for _ in range(num_objects):
        y = np.random.randint(0, height - object_size)
        x = np.random.randint(0, width - object_size)
        image[y:y + object_size, x:x + object_size] = 255
    return image

def save_image(image, directory, filename):
    """
    Save the generated image to a specified directory.
    
    Args:
        image (np.ndarray): The image to save.
        directory (str): The directory to save the image in.
        filename (str): The filename for the image.
    """
    os.makedirs(directory, exist_ok=True)
    img = Image.fromarray(image)
    img.save(os.path.join(directory, filename))

def main():
    num_train = 800
    num_test = 200
    
    # Generate and save training images
    for i in range(num_train):
        dummy_image = generate_dummy_image(num_objects=60)
        save_image(dummy_image, 'train_images', f'train_image_{i:04d}.png')
    
    # Generate and save test images
    for i in range(num_test):
        dummy_image = generate_dummy_image(num_objects=60)
        save_image(dummy_image, 'test_images', f'test_image_{i:04d}.png')

if __name__ == "__main__":
    main()