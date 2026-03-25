import torch
import torch.nn as nn
from einops import einsum

def softmax(x: torch.Tensor, dim: int = 0, temperature: float = 1) -> torch.Tensor:  
    v = x - x.max(dim=dim, keepdim=True)[0]
    
    if temperature > 0 and temperature != 1:
        v/=temperature

    v_exp = v.exp()
    return v_exp / v_exp.sum(dim=dim, keepdim=True)

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None, swiglu=False):
        super().__init__()
        std = (2 / (in_features + out_features)) ** 0.5
        self.weights = nn.parameter.Parameter(
            nn.init.trunc_normal_(
                torch.empty(out_features, in_features, device=device, dtype=dtype),
                mean=0,
                std=std,
                a=-3*std,
                b=3*std
            )
        )
        if(swiglu):
            self.swiglu_weights = nn.parameter.Parameter(
                nn.init.trunc_normal_(
                    torch.empty(out_features, in_features, device=device, dtype=dtype),
                    mean=0,
                    std=std,
                    a=-3*std,
                    b=3*std
                )
            )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "batch ... in_features, out_features in_features -> batch ... out_features")

    
    def swiglu(self, y:torch.Tensor, x:torch.Tensor) -> torch.Tensor:  #-> #TODO: fix the structure
        '''
            y: Linear transformation on the input (forward(x))
            x: input
        '''
        silu = torch.mul(y, torch.sigmoid(y))
        glu = einsum(x, self.swiglu_weights, "batch ... in_features, out_features in_features -> batch ... out_features")
        return torch.mul(silu, glu)
    
class FCN(nn.Module):
    def __init__(self, d_model: int, int_dim: int, device=None, dtype=None):
        super().__init__()
        self.up = Linear(in_features=d_model, out_features=int_dim, device=device, dtype=dtype, swiglu=True)
        self.down = Linear(in_features=int_dim, out_features=d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.up(x)
        up = self.up.swiglu(y=up, x=x)
        down = self.down(up)
        return down
