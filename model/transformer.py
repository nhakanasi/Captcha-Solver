import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import copy
from utils.tool import *
def attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        # Get the target dimensions from scores
        batch_size, num_heads, seq_len_q, seq_len_k = scores.shape
        
        # Handle mask dimensions properly
        original_mask = mask
        
        # Ensure mask has at least 2 dimensions
        while mask.dim() < 2:
            mask = mask.unsqueeze(0)
        
        # Handle different mask shapes
        if mask.dim() == 2:  # [seq_len, seq_len]
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        elif mask.dim() == 3:  # [1, seq_len, seq_len] or [batch, seq_len, seq_len]
            mask = mask.unsqueeze(1)  # Add head dimension
        
        # Now mask should be 4D, but let's double-check
        if mask.dim() != 4:
            # If still not 4D, create a proper 4D mask
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            else:
                # Fallback: create identity mask
                mask = torch.ones(1, 1, seq_len_q, seq_len_k, device=q.device, dtype=torch.bool)
        
        # Now we can safely unpack
        mask_batch, mask_heads, mask_seq_q, mask_seq_k = mask.shape
        
        # If mask sequence length doesn't match scores sequence length, resize mask
        if mask_seq_q != seq_len_q or mask_seq_k != seq_len_k:
            # Create a new mask with the correct size
            if mask_seq_q == mask_seq_k and seq_len_q == seq_len_k:  # Square masks - likely causal
                # Recreate causal mask with correct size
                new_mask = torch.tril(torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool))
                new_mask = new_mask.unsqueeze(0).unsqueeze(0)
            else:
                # Create new mask and copy available values
                new_mask = torch.ones(1, 1, seq_len_q, seq_len_k, device=q.device, dtype=torch.bool)
                min_q = min(mask_seq_q, seq_len_q)
                min_k = min(mask_seq_k, seq_len_k)
                new_mask[:, :, :min_q, :min_k] = mask[:, :, :min_q, :min_k]
            mask = new_mask
        
        # Expand mask to match batch and head dimensions
        mask = mask.expand(batch_size, num_heads, seq_len_q, seq_len_k)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
    
    output = torch.matmul(scores, v)
    return output, scores

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.emb = nn.Embedding(vocab_size,d_model)
    def forward(self, x):
        return self.emb(x)
    
class PostitionalEncoder(nn.Module):
    def __init__(self, d_model, max_len = 1024, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.drop_out = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)

        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos/(10000**(2*i/d_model)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*i+1)/d_model)))
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) 
    
    def forward(self, x):

        x = x*math.sqrt(self.d_model)
        seq_length = x.size(1)

        pe = torch.tensor(self.pe[:,:seq_length, :self.d_model],requires_grad=False)
        
        if x.is_cuda:
            pe.cuda()

        x = x+pe
        x = self.drop_out(x)
        return x
    

class MultiheadAttention(nn.Module):
    def __init__(self, head, d_model, dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model//head
        self.h = head

        self.attn = None

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q,k,v, mask = None):
        b = q.size(0)

        q = self.q_linear(q).view(b, -1, self.h, self.d_k)
        k = self.k_linear(k).view(b, -1, self.h, self.d_k)
        v = self.v_linear(v).view(b, -1, self.h, self.d_k)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores, self.attn = attention(q, k, v, mask, self.dropout)

        concat = scores.transpose(1, 2).contiguous().view(b, -1, self.d_model)
    
        output = self.out(concat)
        return output
    
class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__() 
    
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiheadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x1 = self.norm_1(x)
        x = self.dropout_1(self.attn(x1,x1,x1,mask)) + x
        x1 = self.norm_2(x)
        x = self.dropout_2(self.ff(x1)) + x
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiheadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiheadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x1 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x1, x1, x1, trg_mask))
        x1 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x1, e_outputs, e_outputs, src_mask))
        x1 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x1))
        return x

def clone(module,N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, d_ff = 2048, max_len = 1024, use_embedding=True):
        super().__init__()
        self.N = N
        self.use_embedding = use_embedding
        if use_embedding:
            self.embed = Embedder(vocab_size, d_model)
        self.pe = PostitionalEncoder(d_model, max_len, dropout=dropout)
        self.layers = clone(EncoderLayer(d_model, heads, d_ff, dropout), N)
        self.norm = Norm(d_model)
    
    def forward(self, x, mask):
        if self.use_embedding:
            x = self.embed(x)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x,mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff = 2048, dropout = 0.1):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PostitionalEncoder(d_model, dropout=dropout)
        self.layers = clone(DecoderLayer(d_model, heads, d_ff, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, x, e_out, src_mask = None,trg_mask = None):
        x = x.long()
        x = self.embed(x)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_out, src_mask, trg_mask)
        return self.norm(x)
class Transformer(nn.Module):
    def __init__(self, src_vocab, d_model, N, heads, d_ff_encoder = 2048, d_ff_decoder = 2048, dropout = 0.1, max_len = 1024, use_src_embedding=False):
        super().__init__()
        self.encoder = Encoder(src_vocab,d_model,N,heads,dropout, d_ff_encoder, max_len, use_embedding=use_src_embedding)
        self.decoder = Decoder(src_vocab,d_model,N,heads, d_ff_decoder ,dropout)
        self.out = nn.Linear(d_model,src_vocab)

        
    def forward(self, src, trg, src_mask = None, trg_mask = None):
        e_out  = self.encoder(src, src_mask)
        target = self.decoder(trg, e_out, src_mask, trg_mask)
        out = self.out(target)
        return out

    def forward_encoder(self,src, mask = None):
        src =  self.encoder(src, mask)
        return src
    
    def forward_decoder(self, tgt, memory):
        seq_len = tgt.shape[0]
        tgt_mask = nopeak_mask(seq_len, tgt.device)
        out = self.decoder(tgt, memory, trg_mask=tgt_mask)
        out = out.transpose(0, 1)
        
        return self.out(out), memory

    
    def expand_memory(self, memory, beam_size):
        memory = memory.repeat(1, beam_size, 1)
        return memory

    def get_memory(self, memory, i):
        memory = memory[:, [i], :]
        return memory