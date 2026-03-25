import torch
import numpy

data = np.memmap('fwedutokenized.bin', dtype=np.uint16, mode='r')

print(f"Loaded dataset with {len(data):,} tokens.")

def get_batch(data, batch_size, seq_length, device='cuda'):
    """Grabs a random batch of sequences and targets from the memory-mapped file."""
    
    # Generate random starting indices for our batch
    # We subtract seq_length so we don't accidentally read past the end of the 37GB file
    ix = torch.randint(len(data) - seq_length, (batch_size,))
    
    # Extract the input sequences (x) and target sequences (y)
    x = torch.stack([torch.from_numpy((data[i : i + seq_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + seq_length]).astype(np.int64)) for i in ix])
    
    # Move to GPU if available
    x, y = x.to(device), y.to(device)
    
    return x, y