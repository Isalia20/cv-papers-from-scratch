import torch
from torch import nn
from typing import List


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_block = ConvBlock(in_channels=in_channels, out_channels=out_channels, stride=stride)
        # Convolution for downsampling spatial dimensions and increasing channels for residual connection
        self.downsample_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x_before = x
        x = self.conv_block(x)
        # Residual connection(if channels don't match follow 3.3 section in paper(option B))
        if x_before.shape[-1] != x.shape[-1]:
            x_before = self.downsample_conv(x_before)
        x = x + x_before
        x = self.relu(x)
        return x


class ResnetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks: int, stride_init_block):
        super().__init__()
        self.initial_block = ResBlock(in_channels=in_channels, out_channels=out_channels, stride=stride_init_block)
        self.resnet_blocks = nn.ModuleList([ResBlock(in_channels=out_channels, out_channels=out_channels) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.initial_block(x)
        for layer in self.resnet_blocks:
            x = layer(x)
        return x

class Resnet34(nn.Module):
    def __init__(self, in_channels, out_channels: List[int], out_features: int):
        super().__init__()
        # Initial 7x7 convolution
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = ResnetLayer(in_channels=out_channels[0], out_channels=out_channels[0], num_blocks=3, stride_init_block=1)
        self.layer2 = ResnetLayer(in_channels=out_channels[0], out_channels=out_channels[1], num_blocks=4, stride_init_block=2)
        self.layer3 = ResnetLayer(in_channels=out_channels[1], out_channels=out_channels[2], num_blocks=6, stride_init_block=2)
        self.layer4 = ResnetLayer(in_channels=out_channels[2], out_channels=out_channels[3], num_blocks=3, stride_init_block=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(in_features=512, out_features=out_features)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x


def main():
    x = torch.rand((1, 3, 224, 224))
    model = Resnet34(in_channels=3, out_channels=[64, 128, 256, 512], out_features=10)
    with torch.inference_mode():
        out = model(x)
    print(out.shape)

if __name__ == "__main__":
    main()
