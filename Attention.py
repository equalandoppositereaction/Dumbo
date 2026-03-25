import torch
import torch.nn as nn
from einops import rearrange

#pip install flash-linear-attention

#cannot impliment it cause time shortage
#TODO: write the RDA and GLA from scratch



def softmax(x: torch.Tensor, dim: int = 0, temperature: float = 1) -> torch.Tensor:  
    v = x - x.max(dim=dim, keepdim=True)[0]
    
    if temperature > 0 and temperature != 1:
        v/=temperature

    v_exp = v.exp()
    return v_exp / v_exp.sum(dim=dim, keepdim=True)

class Attention(nn.Module):
    def __init__()
