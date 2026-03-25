import torch
import torch.nn as nn
from einops import rearrange
from Linear import Linear
#pip install flash-linear-attention

#cannot impliment it cause time shortage
#TODO: write the RDA and GLA from scratch



def softmax(x: torch.Tensor, dim: int = 0, temperature: float = 1) -> torch.Tensor:  
    v = x - x.max(dim=dim, keepdim=True)[0]
    
    if temperature > 0 and temperature != 1:
        v/=temperature

    v_exp = v.exp()
    return v_exp / v_exp.sum(dim=dim, keepdim=True)

class GroupedQueryAttention(nn.Module):
    def __init__(self, input_dim, num_heads, num_groups, device=None, dtype=None):
        super(GroupedQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = input_dim // num_heads
        self.query_proj = Linear(input_dim, input_dim, device=device, dtype=dtype)
        self.key_proj = Linear(input_dim, input_dim, device=device, dtype=dtype)
        self.value_proj = Linear(input_dim, input_dim, device=device, dtype=dtype)
        self.out_proj = Linear(input_dim, input_dim, device=device, dtype=dtype)
 
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        queries = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
 
        # Group the queries
        queries = queries.view(batch_size, seq_len, self.num_groups, self.num_heads // self.num_groups, self.head_dim)
        queries = queries.transpose(2, 3).contiguous().view(batch_size, seq_len, self.num_heads, self.head_dim)
 
        # Compute attention scores
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
 
        # Compute attention output
        attn_output = torch.matmul(attn_probs, values)
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)
        return output
