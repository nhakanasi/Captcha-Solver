import torch
from torch import tensor
import numpy as np
from PIL import Image
import numpy as np


def nopeak_mask(size, device):
    # Create lower triangular matrix (1s below diagonal, 0s above)
    mask = torch.tril(torch.ones(size, size, device=device, dtype=torch.bool))
    return mask.unsqueeze(0)  

def create_masks(src, trg, src_pad, trg_pad, device):
    src_mask = tensor((src != src_pad)).unsqueeze(-1)

    if trg is not None:
        trg_mask = (trg != trg_pad).unsqueeze(-1)
        size = trg.size(1)
        np_mask = nopeak_mask(size, device)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask
        
    else:
        trg_mask = None
    return src_mask, trg_mask

def generate_square_subsequent_mask(sz):
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf')."""
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask