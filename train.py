import torch
import numpy as np

data = np.memmap('fwedutokenized.bin', dtype=np.uint16, mode='r')
print('loaded')

