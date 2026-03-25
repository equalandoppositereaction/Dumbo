import torch
import torch.nn as nn
from einops import rearrange



class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        thetas = torch.tensor(theta, device=device).unsqueeze(0).repeat(d_k // 2) #dont fully understand this block saw it in a repo
        j = torch.arange(0, d_k // 2, device=device)
        inv_angles = theta ** (-2 * j / d_k)
        thetas = torch.outer(
            torch.arange(max_seq_len, device=device), inv_angles
        )

        cos, sin = thetas.cos(), thetas.sin()
        #recommended in cs336
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype

        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x_pairs = rearrange(
            x.to(torch.float32), "... seq_len (d_k_half t) -> ... seq_len d_k_half t", t=2
        )

        x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
        row1 = x1 * cos - x2 * sin
        row2 = x2 * cos + x1 * sin
        rotated = torch.stack([row1, row2], dim=-1)

        rotated = rearrange(
            rotated, "... seq_len d_k_half t -> ... seq_len (d_k_half t)", t=2
        )
        return rotated.to(in_type)

