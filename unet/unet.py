from typing import List
import torch
from torch import nn
from torchvision.transforms.functional import center_crop


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        self.pooling_layer = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)), inplace=False)
        conv_output = nn.functional.relu(self.bn2(self.conv2(x)), inplace=False)
        layer_output = self.pooling_layer(conv_output)
        # We return conv_output to then concatenate it later
        return conv_output, layer_output


class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels // 2, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
    
    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)), inplace=False)
        x = nn.functional.relu(self.bn2(self.conv2(x)), inplace=False)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Upsample doesn't reduce the number of channels
        self.up_conv = nn.ConvTranspose2d(in_channels=in_channels, 
                                           out_channels=in_channels,
                                           kernel_size=2, 
                                           stride=2)
        # in_channels is multiplied by 2 due to concatenation before convolution
        self.conv1 = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels // 2)

    def forward(self, x, x_last):
        x = self.up_conv(x) # [1, 512, 56, 56] for example
        x_last = center_crop(x_last, x.shape[-2:]) # Center crop the input from last layer
        x = torch.cat([x_last, x], dim=1) # concatenate on channel dimension
        x = nn.functional.relu(self.bn1(self.conv1(x)), inplace=False)
        x = nn.functional.relu(self.bn2(self.conv2(x)), inplace=False)
        return x
    

class LastUpsampleBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels=in_channels,
                                          out_channels=in_channels,
                                          kernel_size=2,
                                          stride=2)
        self.conv1 = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
    
    def forward(self, x, x_last):
        x = self.up_conv(x)
        x_last = center_crop(x_last, x.shape[-2:])
        x = torch.cat([x_last, x], dim=1)
        x = nn.functional.relu(self.bn1(self.conv1(x)), inplace=False)
        x = nn.functional.relu(self.bn2(self.conv2(x)), inplace=False)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch, out_chs: List[int]):
        super().__init__()
        # [64, 128, 256, 512, 1024]
        # [0, 1, 2, 3, 4]
        self.down_block1 = DownsampleBlock(in_channels=in_ch, out_channels=out_chs[0])
        self.down_block2 = DownsampleBlock(in_channels=out_chs[0], out_channels=out_chs[1])
        self.down_block3 = DownsampleBlock(in_channels=out_chs[1], out_channels=out_chs[2])
        self.down_block4 = DownsampleBlock(in_channels=out_chs[2], out_channels=out_chs[3])
        self.bottleneck_block = BottleNeckBlock(in_channels=out_chs[3], out_channels=out_chs[4])
        self.up_block1 = UpsampleBlock(in_channels=out_chs[3])
        self.up_block2 = UpsampleBlock(in_channels=out_chs[2])
        self.up_block3 = UpsampleBlock(in_channels=out_chs[1])
        # Last upsample block is a class of its own due to the unet architecture
        # not reducing the output channels 4 times before the output
        # 128 -> 64 -> 64
        self.up_block4 = LastUpsampleBlock(in_channels=out_chs[0])
        self.out = nn.Conv2d(in_channels=out_chs[0], out_channels=1, kernel_size=1) #NOTE this is different from unet, for some reason it has 2 channels in output
    
    def forward(self, x):
        conv_out1, x = self.down_block1(x)
        conv_out2, x = self.down_block2(x)
        conv_out3, x = self.down_block3(x)
        conv_out4, x = self.down_block4(x) # conv_out4 is [1, 512, 64, 64]
        x = self.bottleneck_block(x) # 1, 512, 28, 28
        x = self.up_block1(x, conv_out4)
        x = self.up_block2(x, conv_out3)
        x = self.up_block3(x, conv_out2)
        x = self.up_block4(x, conv_out1)
        x = self.out(x)
        return x


def main():
    image = torch.rand((1, 3, 572, 572))
    model = UNet(in_ch=3, out_chs=[64, 128, 256, 512, 1024])
    model.cuda()
    image = image.cuda()
    with torch.inference_mode():
        out = model(image)
    print(out.shape)


if __name__ == "__main__":
    main()
