import torch
from torch import nn
import math

def window_partition(x, window_size):
    B, emb_dim, H, W = x.shape
    # B, emb_dim, H, W -> B, num_windows, window_size, window_size, emb_dim
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, emb_dim)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, window_size, window_size, emb_dim)
    return windows


class WindowedAttention(nn.Module):
    """
    Regular windowed attention
    """
    def __init__(self, emb_dim, num_attn_heads = 8, feature_map_size: int = 56, window_size: int = 7):
        super().__init__()
        self.num_attn_heads = num_attn_heads
        self.emb_dim = emb_dim
        self.emb_dim_per_head = emb_dim // num_attn_heads
        self.window_size = window_size
        self.q_proj = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.k_proj = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.v_proj = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.feature_map_size = feature_map_size
        # Relative position bias table (Eqn. 4)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((window_size * 2 - 1) * (window_size * 2 - 1), num_attn_heads))
        self.initialize_relative_pos_index(window_size)
    
    def initialize_relative_pos_index(self, window_size):
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        mesh = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        mesh = mesh.flatten(1)
        # Calculate distance from each coordinate to all other coordinates, tensor of shape: 2, 49, 49 (for window_size=7)
        relative_coords = mesh[:, :, None] - mesh[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0)
        relative_coords += torch.abs(relative_coords.min())
        # Scale x coordinate as in to avoid (1, 2) coordinate and (2, 1) coordinate colliding
        relative_coords[:, :, 0] = relative_coords[:, :, 0] * (2 * window_size - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        

    def split_num_heads(self, x):
        # Splits windowed tensor into multiple heads
        # B, num_windows, window_size, window_size, emb_dim -> B, num_windows, num_attn_heads, window_size * window_size, emb_dim_per_head 
        B, num_windows, window_size, window_size, emb_dim = x.shape
        return x.reshape(B, num_windows, self.num_attn_heads, window_size * window_size, self.emb_dim_per_head)
    
    def forward(self, x):
        B, L, emb_dim = x.shape
        x = x.reshape(B, emb_dim, self.feature_map_size, self.feature_map_size)
        windows = window_partition(x, window_size=7)
        q = self.q_proj(windows)
        k = self.k_proj(windows)
        v = self.v_proj(windows)

        q = self.split_num_heads(q)
        k = self.split_num_heads(k)
        v = self.split_num_heads(v)
        # # Step 2: Attention score and final scores(with relative positional bias added Eqn. 4)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention_scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.emb_dim_per_head)
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        scores = attention_scores @ v
        return scores.reshape(B, L, emb_dim)
    

class MLPLayer(nn.Module):
    def __init__(self, emb_dim, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(emb_dim, intermediate_size)
        self.gelu = nn.GELU()
        self.output_proj = nn.Linear(intermediate_size, emb_dim)
    
    def forward(self, hidden_states):
        hidden_states = self.gelu(self.dense(hidden_states))
        # Scale it back to original embedding size
        hidden_states = self.output_proj(hidden_states)
        return hidden_states


class PatchMerging(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.reduction = nn.Linear(in_features=emb_dim * 4, out_features=emb_dim * 2)
        self.norm = nn.LayerNorm(emb_dim * 4)

    def forward(self, x):
        # x is shape of [B, H, W, emb_dim]
        # The first patch merging layer concatenates the features of each group of 2 × 2 neighboring patches, 
        # and applies a linear layer on the 4C-dimensional concatenated features.
        # Example explanation of how the below code works:
        # Input:
        # [A][B][C][D]
        # [E][F][G][H]
        # [I][J][K][L]
        # [M][N][O][P]

        # x0: [A][C]
        #     [I][K]

        # x1: [E][G]
        #     [M][O]

        # x2: [B][D]
        #     [J][L]

        # x3: [F][H]
        #     [N][P]
        # Each of these tensors being [B, H / 2, W / 2, emb_dim]
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1) # Shape of [B, emb_dim * 4, H // 2, W // 2], spatial dimensions get downsampled
        x = self.norm(x)
        x = self.reduction(x)
        return x

class SwinBlock(nn.Module):
    """
    Class for Swinblock which in reality is 2 blocks in one. See Fig.3(b) in paper
    to get a good understanding of why this is implemented it this way
    """
    def __init__(self, emb_dim, feature_map_size):
        super().__init__()
        self.norm_pre_attn = nn.LayerNorm(emb_dim)
        self.norm_pre_mlp = nn.LayerNorm(emb_dim)
        self.window_attention = WindowedAttention(emb_dim, num_attn_heads=8, feature_map_size=feature_map_size)
        self.mlp = MLPLayer(emb_dim=emb_dim, intermediate_size=emb_dim * 2)
    
    def forward(self, x):
        h = self.norm_pre_attn(x)
        h = self.window_attention(x)
        # Residual
        h += x
        # # Norm before mlp
        h_mlp = self.norm_pre_mlp(h)
        # # MLP
        h_mlp = self.mlp(h)
        h_mlp += h
        return h


class SwinTransformer(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, stage_1_depth: int, stage_2_depth: int, stage_3_depth: int, stage_4_depth: int, num_classes):
        super().__init__()
        self.emb_dim = embed_dim
        self.patch_partitioner = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        # Stage 1
        self.stage_1_blocks = nn.ModuleList([SwinBlock(emb_dim=embed_dim, feature_map_size=56) for _ in range(stage_1_depth)])
        # Stage 2
        self.stage_2_blocks = nn.ModuleList([SwinBlock(emb_dim=embed_dim * 2, feature_map_size=28) for _ in range(stage_2_depth)])
        self.stage_2_patch_merging = PatchMerging(emb_dim=embed_dim)
        # Stage 3
        self.stage_3_blocks = nn.ModuleList([SwinBlock(emb_dim=embed_dim * 4, feature_map_size=14) for _ in range(stage_3_depth)])
        self.stage_3_patch_merging = PatchMerging(emb_dim=embed_dim * 2)
        # Stage 4
        self.stage_4_blocks = nn.ModuleList([SwinBlock(emb_dim=embed_dim * 8, feature_map_size=7) for _ in range(stage_4_depth)])
        self.stage_4_patch_merging = PatchMerging(emb_dim=embed_dim * 4)
        # Output layers
        self.norm = nn.LayerNorm(embed_dim * 8)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(self.emb_dim * 8, num_classes)
    
    def stage_forward(self, x, block_module_list, patch_merging_module, emb_dim):
        x = x.reshape(x.shape[0], int(math.sqrt(x.shape[1])), int(math.sqrt((x.shape[1]))), emb_dim)
        x = patch_merging_module(x).transpose(1, -1) # B, H, W, emb_dim -> B, emb_dim, H, W
        x = x.flatten(2).transpose(-1, -2)
        for block in block_module_list:
            x = block(x)
        return x
    
    def output_layer(self, x):
        total_elements = x.shape[0] * x.shape[1] * x.shape[2]
        side_length = int(math.sqrt(total_elements // (x.shape[0] * self.emb_dim * 8)))
        x = x.reshape(x.shape[0], self.emb_dim * 8, side_length, side_length)
        return x
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        #TODO fix this organization wtf is this
        # Instead of having patch partitioner first and then linear layer, they are done
        # with one conv, since combining them is just one linear function
        x = self.patch_partitioner(x)
        x = x.flatten(2).transpose(-1, -2)
        # Stage 1 swin block
        for block in self.stage_1_blocks:
            x = block(x)
        # Stage 2
        x = self.stage_forward(x, self.stage_2_blocks, self.stage_2_patch_merging, emb_dim=self.emb_dim)
        # Stage 3
        x = self.stage_forward(x, self.stage_3_blocks, self.stage_3_patch_merging, emb_dim=self.emb_dim * 2)
        # Stage 4
        x = self.stage_forward(x, self.stage_4_blocks, self.stage_4_patch_merging, emb_dim=self.emb_dim * 4)
        # Output layers TODO refactor this!!!
        x = self.output_layer(x)
        x = self.avgpool(x).squeeze(-1).squeeze(-1)
        x = self.head(x)
        return x


def main():
    x = torch.rand((1, 3, 224, 224))
    model = SwinTransformer(in_channels=3, patch_size=4, embed_dim=96, stage_1_depth=2, stage_2_depth=2, stage_3_depth=6, stage_4_depth=2, num_classes=10)
    out = model(x)
    print("OUTPUT SHAPE IS ", out.shape)

if __name__ == "__main__":
    main()
