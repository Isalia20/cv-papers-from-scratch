import torch
from torch import nn
from typing import List
import math


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        expansion = 4
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels * expansion)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        expansion = 4
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_block = ConvBlock(in_channels=in_channels, out_channels=out_channels, stride=stride)
        # Convolution for downsampling spatial dimensions and increasing channels for residual connection
        self.downsample_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * expansion, kernel_size=1, stride=stride)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x_before = x
        x = self.conv_block(x)
        if x_before.shape[1] != x.shape[1]:
            x_before = self.downsample_conv(x_before)

        x = x + x_before
        x = self.relu(x)
        return x


class ResnetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks: int, stride_init_block):
        super().__init__()
        expansion = 4
        self.initial_block = ResBlock(in_channels=in_channels, out_channels=out_channels, stride=stride_init_block)
        self.resnet_blocks = nn.ModuleList([ResBlock(in_channels=out_channels * expansion, out_channels=out_channels) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.initial_block(x)
        for layer in self.resnet_blocks:
            x = layer(x)
        return x

class Resnet50(nn.Module):
    def __init__(self, in_channels, out_channels: List[int]):
        super().__init__()
        # Initial 7x7 convolution
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = ResnetLayer(in_channels=out_channels[0], out_channels=out_channels[0], num_blocks=3, stride_init_block=1)
        self.layer2 = ResnetLayer(in_channels=out_channels[0] * 4, out_channels=out_channels[1], num_blocks=4, stride_init_block=2)
        self.layer3 = ResnetLayer(in_channels=out_channels[1] * 4, out_channels=out_channels[2], num_blocks=6, stride_init_block=2)
        self.layer4 = ResnetLayer(in_channels=out_channels[2] * 4, out_channels=out_channels[3], num_blocks=3, stride_init_block=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
       
       
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_attn_heads):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_attn_heads = num_attn_heads
        self.emb_dim_per_head = self.emb_dim // self.num_attn_heads
    
    def split_multi_head(self, x):
        return x.reshape(x.shape[0], self.num_attn_heads, -1, self.emb_dim_per_head)
    
    def gather_multi_head(self, x):
        return x.reshape(x.shape[0], -1, self.emb_dim)
    
    def forward(self, q, k, v):
        q = self.split_multi_head(q)
        k = self.split_multi_head(k)
        v = self.split_multi_head(v)
        attention_scores = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(self.emb_dim_per_head), dim=-1)
        scores = attention_scores @ v
        # multi head -> one head
        scores = self.gather_multi_head(scores)
        return scores


class MLPLayer(nn.Module):
    def __init__(self, emb_dim, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(emb_dim, intermediate_size)
        self.silu = nn.SiLU(inplace=True)
        self.output_proj = nn.Linear(intermediate_size, emb_dim)
    
    def forward(self, hidden_states):
        hidden_states = self.silu(self.dense(hidden_states))
        # Scale it back to original embedding size
        hidden_states = self.output_proj(hidden_states)
        return hidden_states

 
class EncoderBlock(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.q_proj = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.k_proj = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.v_proj = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.attention = MultiHeadAttention(emb_dim=emb_dim, num_attn_heads=8)
        self.norm_after_attn = nn.LayerNorm(emb_dim)
        self.norm_after_ffn = nn.LayerNorm(emb_dim)
        self.mlp_layer = MLPLayer(emb_dim=emb_dim, intermediate_size=768)
    
    def forward(self, x, pos_emb):
        # Input is of shape [B, HW, emb_dim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # According to Fig. 10 in the paper, which is different from vanilla transformer
        # i.e. positional embeddings are only added to q and k values instead of X before projection
        q += pos_emb
        k += pos_emb
        h = self.attention(q, k, v)
        h += x
        # Normalize
        h_res = self.norm_after_attn(h)
        # FFN part of encoder block
        h = self.mlp_layer(h)
        h += h_res
        h = self.norm_after_ffn(h)
        return h


class Encoder(nn.Module):
    def __init__(self, emb_dim, num_blocks):
        super().__init__()
        self.encoder_block = EncoderBlock(emb_dim=emb_dim)
        self.blocks = nn.ModuleList([EncoderBlock(emb_dim) for _ in range(num_blocks)])
    
    def forward(self, x, pos_emb):
        for layer in self.blocks:
            x = layer(x, pos_emb)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.q_proj_self = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.k_proj_self = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.v_proj_self = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.q_proj_cross = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.k_proj_cross = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.v_proj_cross = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.self_attention = MultiHeadAttention(emb_dim=emb_dim, num_attn_heads=8)
        self.cross_attention = MultiHeadAttention(emb_dim=emb_dim, num_attn_heads=8)
        self.norm_after_self_attn = nn.LayerNorm(emb_dim)
        self.norm_after_cross_attn = nn.LayerNorm(emb_dim)
        self.norm_after_mlp = nn.LayerNorm(emb_dim)
        self.mlp_layer = MLPLayer(emb_dim=emb_dim, intermediate_size=768)

    def forward(self, x, obj_queries, pos_emb, encoder_out):
        # This closely follows Figure 10 in paper
        # Every operation before Multi head attention
        q = self.q_proj_self(x)
        k = self.k_proj_self(x)
        v = self.v_proj_self(x)
        q += obj_queries
        k += obj_queries
        h = self.self_attention(q, k, v)
        h += x
        h = self.norm_after_self_attn(h)

        # Pre multi head attention(cross attention)
        q_cross = self.q_proj_cross(h) + obj_queries
        k_cross = self.k_proj_cross(encoder_out) + pos_emb # K is output from Encoder + pos embeddings
        v_cross = self.v_proj_cross(encoder_out)
        h_cross = self.cross_attention(q_cross, k_cross, v_cross)
        h_cross += h
        h_cross = self.norm_after_cross_attn(h_cross)

        # FFN
        out = self.mlp_layer(h_cross)
        out += h_cross
        out = self.norm_after_mlp(out)
        return out


class Decoder(nn.Module):
    def __init__(self, emb_dim, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([DecoderBlock(emb_dim) for _ in range(num_blocks)])
    
    def forward(self, x, obj_queries, pos_emb, encoder_out):
        for block in self.blocks:
            x = block(x, obj_queries, pos_emb, encoder_out)
        return x


class DETR(nn.Module):
    def __init__(self, emb_dim, backbone_feature_map_size, num_encoder_blocks, num_decoder_blocks, num_classes):
        super().__init__()
        self.emb_dim = emb_dim
        self.expansion = 4
        self.backbone_out_channels = [64, 128, 256, 512]
        self.backbone = Resnet50(in_channels=3, out_channels=self.backbone_out_channels)
        # Convolution for making the output of backbone correspond to emb_dim channels
        self.emb_dim_conv = nn.Conv2d(in_channels=self.backbone_out_channels[-1] * self.expansion, out_channels=emb_dim, kernel_size=1)
        self.pos_emb = nn.Parameter(torch.rand((1, backbone_feature_map_size * backbone_feature_map_size, emb_dim)))
        self.object_queries = nn.Parameter(torch.rand((1, 100, emb_dim)))
        self.encoder = Encoder(emb_dim=emb_dim, num_blocks=num_encoder_blocks)
        self.decoder = Decoder(emb_dim=emb_dim, num_blocks=num_decoder_blocks)
        self.cls_ffn = nn.Linear(in_features=emb_dim, out_features=num_classes + 1)
        self.bbox_ffn = nn.Linear(in_features=emb_dim, out_features=4)
    
    def flatten(self, x):
        # Transforms input from a shape of B, emb_dim, H, W -> B, HW, emb_dim
        return x.reshape((x.shape[0], -1, self.emb_dim))
    
    def forward(self, x):
        # high level outline
        x = self.backbone(x) # 2d feature map
        x = self.emb_dim_conv(x)
        # Flatten and add pos emb
        x = self.flatten(x)
        x = x + self.pos_emb
        # # Encode the image features via vision transformer
        x = self.encoder(x, self.pos_emb)
        x = self.decoder(self.object_queries, self.object_queries, self.pos_emb, x)
        
        # Final prediction
        cls_out = self.cls_ffn(x)
        reg_out = self.bbox_ffn(x)
        return cls_out, reg_out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand((1, 3, 224, 224))
    model = DETR(emb_dim=256, backbone_feature_map_size=x.shape[-1] // 32, num_encoder_blocks=6, num_decoder_blocks=6, num_classes=80)
    x = x.to(device)
    model.to(device)
    out = model(x)
    print(out[0].shape)
    print(out[1].shape)

if __name__ == "__main__":
    main()
