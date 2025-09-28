import torch.nn as nn
from model.transformer import Transformer
from model.vgg import VGG_16, Vgg
from utils.tool import create_masks
import torch
class OCR_model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.cnn = Vgg(ss= [[2, 2],[2, 2],[2, 1],[2, 1],[1, 1]], ks = [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]], hidden=512)
        # Use transformer without source embeddings since CNN outputs continuous features
        self.transformer = Transformer(vocab_size, 256, 6, 8, max_len=1024, use_src_embedding=False)
        self.project = nn.Linear(512,256)
        self.vocab_size = vocab_size
    
    
    def create_tgt_mask(self, size, device):
        """Create a causal mask for the target sequence"""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask == 0

    def forward(self, img, tgt_input):
        src = self.cnn(img)
        src = src.transpose(0, 1)
        # Create dummy src tokens (since CNN output doesn't need padding masks for now)
        # For simplicity, create masks assuming no padding in src and standard padding in target
        device = img.device
        src = self.project(src)
        if tgt_input.dtype != torch.long:
            tgt_input = tgt_input.long()

        src_mask = None  # No masking for CNN features
        trg_mask = self.create_tgt_mask(tgt_input.size(-1), tgt_input.device) # Will be handled in transformer if needed
        if tgt_input.dtype != torch.long:
            tgt_input = tgt_input.long()
        
        
        output = self.transformer(src, tgt_input, src_mask, trg_mask)
        
        return output