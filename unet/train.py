import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import jaccard_score
import torch.nn.init as init
from unet import UNet
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F



class DummyDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = img_dir.replace("images", "masks")
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace("image", "mask"))
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        mask = (mask > 0).float()[[0], ...]
        return image, mask


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for inputs, targets in tqdm(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = F.interpolate(outputs, size=(572, 572), mode='bilinear', align_corners=False)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            outputs = F.interpolate(outputs, size=(572, 572), mode='bilinear', align_corners=False)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            
            all_preds.append(outputs.sigmoid().cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    return running_loss / len(test_loader), all_targets, all_preds


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = DummyDataset('train_images', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, persistent_workers=True)
    
    test_dataset = DummyDataset('test_images', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4, persistent_workers=True)
    
    model = UNet(in_ch=3, out_chs=[64, 128, 256, 512, 1024])
    model.apply(init_weights)
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    num_epochs = 50
    
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        loss, targets, predictions = evaluate(model, test_loader, criterion, device)
        
        iou = jaccard_score((targets > 0.5).flatten(), (predictions > 0.5).flatten())
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, IoU: {iou:.4f}")
        
        img, mask = train_dataset[0]
        img = img.to("cuda")
        mask = mask.to("cuda")
        output = (model(img.unsqueeze(0)).sigmoid() > 0.5).float()
        to_pil_image(output.squeeze(0)).save("segmented_image.jpg")

    print("Training completed.")
    
    torch.save(model.state_dict(), 'unet_model.pth')

if __name__ == "__main__":
    main()
