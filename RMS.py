import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.gain = nn.Parameter(
            nn.init.ones_(torch.empty(d_model, device=device, dtype=dtype))
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps) 
        result = self.gain * x / rms

        return result.to(in_dtype)