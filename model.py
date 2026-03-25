from block import Block
from RMS import RMSNorm
from Linear import Linear
from Embedding import Embedding
import torch
import torch.nn as nn

class Dumbo(nn.Module):
    def __init__(self, num_layers: int, vocab_size: int, d_model: int, fcn_dim: int, num_heads: int, num_groups: int,  device=None, dtype=None):
        super().__init__()
        self.embed = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)

        self.blocks = nn.ModuleList([
            Block(d_model=d_model, num_heads=num_heads, num_groups=num_groups, fcn_dim=fcn_dim, device=device, dtype=dtype) 
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype, swiglu=False)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids)
        
        for block in self.blocks:                                   #-> wow prettier
            x = block(x)

        output = self.lm_head(self.norm(x))
        #out_prob = softmax(output)
        return output
        