import torch
import torch.nn as nn
from einops import rearrange
from Linear import Linear
from positional_embedding import RoPE
#TODO: write RDA and GLA 



def softmax(x: torch.Tensor, dim: int = 0, temperature: float = 1) -> torch.Tensor:  
    v = x - x.max(dim=dim, keepdim=True)[0]
    
    if temperature > 0 and temperature != 1:
        v/=temperature

    v_exp = v.exp()
    return v_exp / v_exp.sum(dim=dim, keepdim=True)

class GQA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_groups: int, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = d_model // num_heads
        self.query_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.key_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.value_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RoPE(device=device)
        #self.position_ids = torch.arange(max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        queries = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        position_ids = torch.arange(seq_len, device=self.query_proj.weight.device).unsqueeze(0).expand(batch_size, -1)
        queries = self.rope(x=queries, token_positions=position_ids)                                                       #->apply RoPE
        keys = self.rope(x=keys, token_positions=position_ids)

        queries = queries.view(batch_size, seq_len, self.num_groups, self.num_heads // self.num_groups, self.head_dim)
        queries = queries.transpose(2, 3).contiguous().view(batch_size, seq_len, self.num_heads, self.head_dim)            #-> group the queries
 
        
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = softmax(attn_scores, dim=-1)
 
        attn_output = torch.matmul(attn_probs, values)
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)
        return output
