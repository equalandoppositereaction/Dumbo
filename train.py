import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import wandb

import model as model_module

data = np.fromfile('fwedutokenized.bin', dtype=np.uint16)
TOTAL_TOKENS = len(data)


BATCH_SIZE = 8
CONTEXT_LENGTH = 1024
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 2000
MAX_STEPS = None  
MIN_LR_RATIO = 0.1  
GRAD_CLIP_NORM = 1.0
BETAS = (0.9, 0.95)

WANDB_API_KEY = "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"  
WANDB_PROJECT = "fine-dumbo"
WANDB_RUN_NAME = None

MODEL_CFG = {
    'num_layers': 16,
    'vocab_size': 40960,  # fill from tokenizer
    'd_model': 1024,
    'fcn_dim': 3584,
    'num_heads': 16,
    'num_groups': 2,
    'device': 'cuda',
    'dtype': torch.bfloat16,
}

def get_batch(x, batch_size, context_length, device):

    idx = torch.randint(0, len(x)-context_length-1, (batch_size,))
    
    inputs = torch.stack([torch.tensor(x[i : i + context_length], dtype=torch.long) for i in idx])
    targets = torch.stack([torch.tensor(x[i + 1 : i + context_length + 1], dtype=torch.long) for i in idx])
    
    inputs, targets = inputs.to(device), targets.to(device)
    
    return inputs, targets

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration

def build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> LambdaLR:
    min_lr_ratio = min_lr_ratio if min_lr_ratio is not None else 0.1

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (cosine * (1 - min_lr_ratio)) + min_lr_ratio  # post-anneal hold at min_lr_ratio

    return LambdaLR(optimizer, lr_lambda)


def train():
    tokens_per_step = BATCH_SIZE * CONTEXT_LENGTH
    total_steps = MAX_STEPS or math.ceil(TOTAL_TOKENS / tokens_per_step)

    cfg = MODEL_CFG.copy()
    cfg['device'] = cfg['device'] or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    net = model_module.Dumbo(**cfg)
    net.to(cfg['device'])
    net.train()

    optimizer = AdamW(net.parameters(), lr=LEARNING_RATE, betas=BETAS, weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, warmup_steps=WARMUP_STEPS, total_steps=total_steps, min_lr_ratio=MIN_LR_RATIO)

    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
        wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config={
            "batch_size": BATCH_SIZE,
            "context_length": CONTEXT_LENGTH,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "min_lr_ratio": MIN_LR_RATIO,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "betas": BETAS,
            "total_tokens": TOTAL_TOKENS,
            "model_cfg": MODEL_CFG,
        })
    else:
        wandb.init(mode="disabled")

    for step in range(total_steps):
        inputs, targets = get_batch(data, BATCH_SIZE, CONTEXT_LENGTH, cfg['device'])
        optimizer.zero_grad(set_to_none=True)
        logits = net(inputs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        loss.backward()
        if GRAD_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP_NORM)

        optimizer.step()
        scheduler.step()

        if step % 10 == 0:
            tokens_done = (step + 1) * tokens_per_step
            lr = scheduler.get_last_lr()[0]
            wandb.log({"loss": loss.item(), "lr": lr, "tokens": tokens_done, "step": step})
            print(f"step={step} loss={loss.item():.4f} lr={lr:.6f} tokens={tokens_done}")


if __name__ == "__main__":
    train()
