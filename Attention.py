import torch
import torch.nn as nn
import torch.nn.functional as F
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
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        if num_heads % num_groups != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_groups ({num_groups})")
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = d_model // num_heads
        self.query_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.key_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.value_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RoPE(device=device)
        #self.position_ids = torch.arange(max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
 
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        queries = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
 
        position_ids = torch.arange(seq_len, device=self.query_proj.weights.device).unsqueeze(0).expand(batch_size, -1)
        cos = self.rope.cos[position_ids].unsqueeze(2)  # (batch, seq_len, 1, d_k/2)
        sin = self.rope.sin[position_ids].unsqueeze(2)  # (batch, seq_len, 1, d_k/2)

        queries = self.rope(x=queries, token_positions=None, cos=cos, sin=sin)                                                       #->apply RoPE
        keys = self.rope(x=keys, token_positions=None, cos=cos, sin=sin)

        queries = queries.view(batch_size, seq_len, self.num_groups, self.num_heads // self.num_groups, self.head_dim)
        queries = queries.transpose(2, 3).contiguous().view(batch_size, seq_len, self.num_heads, self.head_dim)            #-> group the queries

        keys = keys.view(batch_size, seq_len, self.num_groups, self.num_heads // self.num_groups, self.head_dim).mean(dim=3)
        values = values.view(batch_size, seq_len, self.num_groups, self.num_heads // self.num_groups, self.head_dim).mean(dim=3)
        keys = keys.repeat_interleave(self.num_heads // self.num_groups, dim=2)
        values = values.repeat_interleave(self.num_heads // self.num_groups, dim=2)
 
        queries = queries.permute(0, 2, 1, 3)  # (batch, heads, seq, head_dim)
        keys = keys.permute(0, 2, 1, 3)        # (batch, heads, seq, head_dim)
        values = values.permute(0, 2, 1, 3)    # (batch, heads, seq, head_dim)

        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch, heads, seq, seq)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask.to(attn_scores.device), float("-inf"))
        attn_probs = softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_probs, values)  # (batch, heads, seq, head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)
        return output
