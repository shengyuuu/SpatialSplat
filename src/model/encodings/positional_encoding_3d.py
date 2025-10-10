import math
import torch
from torch import nn
from jaxtyping import Float
from einops import rearrange


class PositionalEncoding3D(nn.Module):
    def __init__(self, 
                 num_pos_feats: int = 128,
                 temperature: int = 10000,
                 embed_dim: int = 3):
        super().__init__()
        
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.embed_dim = embed_dim
        
    def forward(self, pos: Float[torch.Tensor, "batch d_in h w"]
                ) -> Float[torch.Tensor, "batch hw d_out"]:
        pos = rearrange(pos, "batch c h w -> batch h w c")
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t
        pos_z = pos[..., 2, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
        if self.embed_dim == 3:
            posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
        if self.embed_dim == 2:
            posemb = torch.cat((pos_y, pos_x), dim=-1)
        posemb = rearrange(posemb, "batch h w c -> batch (h w) c")
        return posemb
    
    @property
    def d_out(self):
        return self.num_pos_feats * self.embed_dim
