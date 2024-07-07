import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from torchvision.transforms.functional import resize
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import jaccard_score


class DummyDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        mask = resize(image, size=(388, 388), antialias=True)
        mask = (mask > 0).int() * 255
        return image, mask


import matplotlib.pyplot as plt
import numpy as np

def visualize_image_and_mask(image, mask):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original image
    ax1.imshow(image.permute(1, 2, 0))  # Change from (C, H, W) to (H, W, C)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot mask
    mask_display = mask.permute(1, 2, 0).numpy().astype(np.uint8)
    ax2.imshow(mask_display, cmap='viridis')
    ax2.set_title('Mask')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig("some_fig.png")

transform = transforms.Compose([
        transforms.Resize((572, 572), antialias=True),
        transforms.ToTensor(),
    ])
# Usage example:
dataset = DummyDataset('train_images', transform=transform)
image, mask = dataset[0]  # Get the first item
visualize_image_and_mask(image, mask)
