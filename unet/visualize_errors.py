import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from train import DummyDataset
from unet import UNet
import matplotlib.pyplot as plt
import numpy as np


def visualize_result(image, mask, prediction, loss, index):
    print(image.shape)
    print(mask.shape)
    print(prediction.shape)
    print(mask)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    ax2.imshow(image)
    ax2.imshow(mask, alpha=0.3, cmap='Greens')
    ax2.imshow(prediction, alpha=0.3, cmap='Reds')
    ax2.set_title(f"Mask (Green) & Prediction (Red)\nLoss: {loss:.4f}")
    ax2.axis('off')
    
    plt.suptitle(f"Sample {index + 1}")
    plt.tight_layout()
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_ch=3, out_chs=[64, 128, 256, 512, 1024])
    model.load_state_dict(torch.load("unet/best_unet_model.pth"))
    model.to(device)
    model.eval()


    transform = transforms.Compose([
        transforms.Resize((572, 572), antialias=True),
        transforms.ToTensor(),
    ])
    test_dataset = DummyDataset('unet/test_images', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    images = []
    masks = []
    predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            losses.append(loss.item())
            images.append(inputs.cpu().squeeze().permute(1, 2, 0).numpy())
            masks.append(targets.cpu().squeeze().numpy())
            predictions.append(torch.sigmoid(outputs).cpu().squeeze().numpy() > 0.5)

    sorted_indices = np.argsort(losses)[::-1]
    N = 5
    for i in range(N):
        index = sorted_indices[i]
        visualize_result(images[index], masks[index], predictions[index], losses[index], i)

if __name__ == "__main__":
    main()
