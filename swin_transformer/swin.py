import torch
from torch import nn
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def window_partition(x, window_size):
    B, emb_dim, H, W = x.shape
    # B, emb_dim, H, W -> B, num_windows, window_size, window_size, emb_dim
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, emb_dim)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, window_size, window_size, emb_dim)
    return windows

def window_unpartition(x, emb_dim, feature_map_size):
    """
    Reverse the window partition, includes reversing the multi head split as well
    """
    # X of shape [B, num_windows, num_attn_heads, feature_map_size * feature_map_size, emb_dim_per_head] -> [B, emb_dim, feature_map_size, feature_map_size]
    return x.reshape(x.shape[0], emb_dim, feature_map_size, feature_map_size)


class SwinAttention(nn.Module):
    """
    Swin Attention
    """
    def __init__(self, emb_dim, num_attn_heads = 8, feature_map_size: int = 56, window_size: int = 7, shift_windows: bool = True): #TODO make this true
        super().__init__()
        self.shift_windows = shift_windows
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
        
        self.shift_size = self.window_size // 2
        self.shifted_window_mask = torch.zeros((1)).to(DEVICE)
        if self.shift_windows:
            self.shifted_window_mask = self.get_sw_attn_mask().to(DEVICE)
        self.shifted_window_mask.requires_grad_(False)
    
    def get_sw_attn_mask(self):
        # Attention mask if we have shifted windows
        H, W = self.feature_map_size, self.feature_map_size
        img_mask = torch.zeros((1, 1, H, W))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, :, h, w] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # num_windows, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask.unsqueeze(0).unsqueeze(2)
    
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
        self.relative_position_index = relative_position_index
        self.relative_position_index.requires_grad_(False)
        

    def split_num_heads(self, x):
        # Splits windowed tensor into multiple heads
        # B, num_windows, window_size, window_size, emb_dim -> B, num_windows, num_attn_heads, window_size * window_size, emb_dim_per_head 
        B, num_windows, window_size, window_size, emb_dim = x.shape
        return x.reshape(B, num_windows, self.num_attn_heads, window_size * window_size, self.emb_dim_per_head)
    
    def forward(self, x):
        B, L, emb_dim = x.shape
        x = x.reshape(B, emb_dim, self.feature_map_size, self.feature_map_size)
        if self.shift_windows:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))
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
        attention_scores += self.shifted_window_mask
        attention_scores = torch.softmax(attention_scores, dim=-1)
        scores = attention_scores @ v
        # Unshift the feature map if shifted
        if self.shift_windows:
            scores = window_unpartition(scores, self.emb_dim, self.feature_map_size)
            scores = torch.roll(scores, shifts=(self.shift_size, self.shift_size), dims=(-2, -1))
        scores = scores.reshape(B, L, emb_dim)
        return scores
    

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
    def __init__(self, emb_dim, feature_map_size, shift_windows):
        super().__init__()
        self.norm_pre_attn = nn.LayerNorm(emb_dim)
        self.norm_pre_mlp = nn.LayerNorm(emb_dim)
        self.swin_attention = SwinAttention(emb_dim, num_attn_heads=8, feature_map_size=feature_map_size, shift_windows=shift_windows)
        self.mlp = MLPLayer(emb_dim=emb_dim, intermediate_size=emb_dim * 2)
    
    def forward(self, x):
        h = self.norm_pre_attn(x)
        h = self.swin_attention(x)
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
        self.stage_1_blocks = nn.ModuleList([SwinBlock(emb_dim=embed_dim, feature_map_size=56, shift_windows=bool(i % 2)) for i in range(stage_1_depth)])
        # Stage 2
        self.stage_2_blocks = nn.ModuleList([SwinBlock(emb_dim=embed_dim * 2, feature_map_size=28, shift_windows=bool(i % 2)) for i in range(stage_2_depth)])
        self.stage_2_patch_merging = PatchMerging(emb_dim=embed_dim)
        # Stage 3
        self.stage_3_blocks = nn.ModuleList([SwinBlock(emb_dim=embed_dim * 4, feature_map_size=14, shift_windows=bool(i % 2)) for i in range(stage_3_depth)])
        self.stage_3_patch_merging = PatchMerging(emb_dim=embed_dim * 2)
        # Stage 4
        self.stage_4_blocks = nn.ModuleList([SwinBlock(emb_dim=embed_dim * 8, feature_map_size=7, shift_windows=bool(i % 2)) for i in range(stage_4_depth)])
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
