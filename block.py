#all imports are in Linear.py
import Linear
import RMS
import Attention
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_groups: int, fcn_dim: int, device=None, dtype=None):
        super().__init__()
        self.a_norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attention = GQA(d_model=d_model, num_heads=num_heads, num_groups=num_groups, device=device, dtype=dtype)

        self.f_norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.fcn = FCN(d_model=d_model, int_dim=fcn_dim, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.a_norm(x)                       #-> normalize input
        attn_out = self.attention(x)             #-> pass through attn block

        attn_out += residual                     #-> add residual

        attn_out = self.f_norm(attn_out)         #-> normalize the attn output + residual
        fcn = self.fcn(attn_out)                 #-> pass it through the fcn

        return fcn + attn_out                    #-> return the fcn output with residual