import torch
from torch import nn

class SwinBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
    
    def forward(self, x):
        pass


class SwinTransformer(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.patch_partitioner = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # Instead of having patch partitioner first and then linear layer, they are done
        # with one conv, since combining them is just one linear function
        x = self.patch_partitioner(x)
        x = x.flatten(2).transpose(-1, -2)
        return x


def main():
    x = torch.rand((1, 3, 224, 224))
    model = SwinTransformer(in_channels=3, patch_size=4, embed_dim=96)
    out = model(x)
    print("OUTPUT SHAPE IS ", out.shape)

if __name__ == "__main__":
    main()
