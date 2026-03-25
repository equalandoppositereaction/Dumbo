import block
import Embedding
import torch
import torch.nn as nn

class Dumbo(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, fcn_dim: int, num_heads: int,num_groups: int,  device=None, dtype=None):
        super().__init__()
        self.embed = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.block1 = Block(d_model=d_model, num_heads=num_heads, num_groups=num_groups, fcn_dim=fcn_dim, device=device, dtype=dtype) #-> just use module bro wtf 
        self.block2 = Block(d_model=d_model, num_heads=num_heads, num_groups=num_groups, fcn_dim=fcn_dim, device=device, dtype=dtype)
        self.block3 = Block(d_model=d_model, num_heads=num_heads, num_groups=num_groups, fcn_dim=fcn_dim, device=device, dtype=dtype)
        self.block4 = Block(d_model=d_model, num_heads=num_heads, num_groups=num_groups, fcn_dim=fcn_dim, device=device, dtype=dtype)
        self.block5 = Block(d_model=d_model, num_heads=num_heads, num_groups=num_groups, fcn_dim=fcn_dim, device=device, dtype=dtype)
        self.block6 = Block(d_model=d_model, num_heads=num_heads, num_groups=num_groups, fcn_dim=fcn_dim, device=device, dtype=dtype)
        self.block7 = Block(d_model=d_model, num_heads=num_heads, num_groups=num_groups, fcn_dim=fcn_dim, device=device, dtype=dtype)
        self.block8 = Block(d_model=d_model, num_heads=num_heads, num_groups=num_groups, fcn_dim=fcn_dim, device=device, dtype=dtype)
        self.block9 = Block(d_model=d_model, num_heads=num_heads, num_groups=num_groups, fcn_dim=fcn_dim, device=device, dtype=dtype)
        self.block10 = Block(d_model=d_model, num_heads=num_heads, num_groups=num_groups, fcn_dim=fcn_dim, device=device, dtype=dtype)
        self.block11 = Block(d_model=d_model, num_heads=num_heads, num_groups=num_groups, fcn_dim=fcn_dim, device=device, dtype=dtype)
        self.block12 = Block(d_model=d_model, num_heads=num_heads, num_groups=num_groups, fcn_dim=fcn_dim, device=device, dtype=dtype)
        self.block13 = Block(d_model=d_model, num_heads=num_heads, num_groups=num_groups, fcn_dim=fcn_dim, device=device, dtype=dtype)
        self.block14 = Block(d_model=d_model, num_heads=num_heads, num_groups=num_groups, fcn_dim=fcn_dim, device=device, dtype=dtype)
        self.block15 = Block(d_model=d_model, num_heads=num_heads, num_groups=num_groups, fcn_dim=fcn_dim, device=device, dtype=dtype)
        self.block16 = Block(d_model=d_model, num_heads=num_heads, num_groups=num_groups, fcn_dim=fcn_dim, device=device, dtype=dtype)
        self.norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype, swiglu=False)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embed = self.embed(token_ids)
        y = self.block1(embed)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)
        y = self.block6(y)
        y = self.block7(y)
        y = self.block8(y)
        y = self.block9(y)
        y = self.block10(y)
        y = self.block11(y)
        y = self.block12(y)
        y = self.block13(y)
        y = self.block14(y)
        y = self.block15(y)
        y = self.block16(y)
        output = self.lm_head(self.norm(y))
        out_prob = softmax(output)
        return output
        