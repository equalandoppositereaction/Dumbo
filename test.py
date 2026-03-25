import os

# Workaround for duplicate OpenMP runtime initialization on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from fla.layers import GatedDeltaNet
bs, num_heads, seq_len, hidden_size = 16, 4, 2048, 512
gated_deltanet = GatedDeltaNet(hidden_size=hidden_size, num_heads=num_heads, mode='chunk').bfloat16()
print(gated_deltanet)