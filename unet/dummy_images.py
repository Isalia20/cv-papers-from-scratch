"""
Module for generating dummy images with various shapes and colors, along with their masks, for training unet model
"""
import numpy as np
import os
from PIL import Image, ImageDraw

def generate_random_color():
    """Generate a random RGB color."""
    return tuple(np.random.randint(0, 256, 3))

def draw_random_shape(draw, mask_draw, x, y, max_size, shape_id):
    """Draw a random shape (rectangle, ellipse, or triangle) with a random color and its mask."""
    shape_type = np.random.choice(['rectangle', 'ellipse', 'triangle'])
    color = generate_random_color()
    # size = np.random.randint(10, 30)
    size = 50

    if shape_type == 'rectangle':
        draw.rectangle([x, y, x + size, y + size], fill=color)
        mask_draw.rectangle([x, y, x + size, y + size], fill=shape_id)
    elif shape_type == 'ellipse':
        draw.ellipse([x, y, x + size, y + size], fill=color)
        mask_draw.ellipse([x, y, x + size, y + size], fill=shape_id)
    else:  # triangle
        points = [
            (x, y + size),
            (x + size // 2, y),
            (x + size, y + size)
        ]
        draw.polygon(points, fill=color)
        mask_draw.polygon(points, fill=shape_id)

def generate_dummy_image_and_mask(size=(572, 572), num_objects=5, max_object_size=120):
    """
    Generate a dummy RGB image with various colored shapes scattered randomly, and its corresponding mask.
    
    Args:
        size (tuple): Size of the generated image (width, height).
        num_objects (int): Number of objects to scatter in the image.
        max_object_size (int): Maximum size of each object.
        
    Returns:
        tuple: (PIL.Image, PIL.Image) Generated dummy image and its mask.
    """
    image = Image.new('RGB', size, color='white')
    mask = Image.new('L', size, color=0)
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)
    
    for i in range(num_objects):
        x = np.random.randint(0, size[0] - max_object_size)
        y = np.random.randint(0, size[1] - max_object_size)
        draw_random_shape(draw, mask_draw, x, y, max_object_size, i + 1)
    
    return image, mask

def save_image(image, directory, filename):
    """
    Save the generated image to a specified directory.
    
    Args:
        image (PIL.Image): The image to save.
        directory (str): The directory to save the image in.
        filename (str): The filename for the image.
    """
    os.makedirs(directory, exist_ok=True)
    image.save(os.path.join(directory, filename))

def main():
    num_train = 50
    num_test = 10
    
    # Generate and save training images and masks
    for i in range(num_train):
        dummy_image, mask = generate_dummy_image_and_mask(num_objects=20, max_object_size=80)
        save_image(dummy_image, 'train_images', f'train_image_{i:04d}.png')
        save_image(mask, 'train_masks', f'train_mask_{i:04d}.png')
    
    # Generate and save test images and masks
    for i in range(num_test):
        dummy_image, mask = generate_dummy_image_and_mask(num_objects=20, max_object_size=80)
        save_image(dummy_image, 'test_images', f'test_image_{i:04d}.png')
        save_image(mask, 'test_masks', f'test_mask_{i:04d}.png')

if __name__ == "__main__":
    main()