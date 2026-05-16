"""
Hierarchical note model: HierarchicalMusicGPT + BarEncoder.

HierarchicalMusicGPT is a drop-in for MusicGPT — existing checkpoints load via
strict=False.  When memory=None the model behaves identically to the original,
so generate.py works correctly even before bar_encoder.pt is trained.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import MusicGPT   # reuse MusicGPT only for its config constants


# ── BarEncoder ────────────────────────────────────────────────────────────────

class _BidirBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        return x + self.ff(self.ln2(x))


class BarEncoder(nn.Module):
    def __init__(self, vocab_size=153, d_enc=128, n_heads=4, n_layers=3,
                 max_len=200, n_summary=4, note_d_model=384):
        super().__init__()
        self.n_summary = n_summary
        self.max_len   = max_len
        self.tok_emb   = nn.Embedding(vocab_size, d_enc)
        self.pos_emb   = nn.Embedding(max_len + n_summary, d_enc)
        self.summary   = nn.Parameter(torch.zeros(1, n_summary, d_enc))
        nn.init.normal_(self.summary, std=0.02)
        self.blocks    = nn.ModuleList([_BidirBlock(d_enc, n_heads) for _ in range(n_layers)])
        self.ln_f      = nn.LayerNorm(d_enc)
        self.proj      = nn.Linear(d_enc, note_d_model)

    def forward(self, bar_tokens):
        B, T = bar_tokens.shape
        T    = min(T, self.max_len)
        x    = self.tok_emb(bar_tokens[:, :T])
        summ = self.summary.expand(B, -1, -1)
        x    = torch.cat([summ, x], dim=1)
        x    = x + self.pos_emb(torch.arange(x.size(1), device=x.device).unsqueeze(0))
        for blk in self.blocks:
            x = blk(x)
        return self.proj(self.ln_f(x)[:, :self.n_summary])


# ── HierarchicalMusicGPT ──────────────────────────────────────────────────────

class _BlockWithCrossAttn(nn.Module):
    """Same ln1/ln2/attn/ff keys as models._Block — loads from existing checkpoint."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )
        self.ln_cross   = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)

    def forward(self, x, causal_mask, memory=None, memory_key_mask=None):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + h
        if memory is not None:
            h = self.ln_cross(x)
            h, _ = self.cross_attn(h, memory, memory,
                                   key_padding_mask=memory_key_mask,
                                   need_weights=False)
            x = x + h
        return x + self.ff(self.ln2(x))


class HierarchicalMusicGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, seq_len, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList(
            [_BlockWithCrossAttn(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        causal = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', causal, persistent=False)

    def forward(self, x, targets=None, loss_mask=None, memory=None, memory_key_mask=None):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        h    = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        mask = self.causal_mask[:T, :T]
        for blk in self.blocks:
            h = blk(h, mask, memory=memory, memory_key_mask=memory_key_mask)
        logits = self.head(self.ln_f(h))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1), ignore_index=0, reduction='none')
        if loss_mask is not None:
            denom = loss_mask.reshape(-1).sum().clamp(min=1)
            loss  = (loss * loss_mask.reshape(-1)).sum() / denom
        else:
            loss = loss[(targets != 0).reshape(-1)].mean()
        return logits, loss


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_hierarchical_note_model(path, device):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = HierarchicalMusicGPT(**ckpt['config']).to(device)
    state = {k: v for k, v in ckpt['state'].items() if k != 'causal_mask'}
    missing, _ = model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_bar_encoder(path, device):
    ckpt    = torch.load(path, map_location=device, weights_only=False)
    encoder = BarEncoder(**ckpt['config']).to(device)
    encoder.load_state_dict(ckpt['state'])
    encoder.eval()
    return encoder


# ── Inference memory builder (single-sample, not batched) ────────────────────

@torch.no_grad()
def build_inference_memory(bar_encoder, bar_history, device, max_history=16):
    """
    Encode bar_history into a (1, H*n_summary, D) memory tensor for inference.
    Returns (None, None) when bar_encoder is None or history is empty —
    cross-attention is then skipped and the model behaves like the original.
    """
    if bar_encoder is None or not bar_history:
        return None, None

    bars      = bar_history[-max_history:]
    n_summary = bar_encoder.n_summary
    D         = bar_encoder.proj.out_features
    H         = len(bars)
    ML        = max_history * n_summary

    memory       = torch.zeros(1, ML, D, device=device)
    mem_key_mask = torch.ones(1, ML, dtype=torch.bool, device=device)

    max_bar_len = min(max(len(b) for b in bars), bar_encoder.max_len)
    bar_batch   = torch.zeros(H, max_bar_len, dtype=torch.long, device=device)
    for i, bar in enumerate(bars):
        t = bar[:max_bar_len]
        bar_batch[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)

    embs = bar_encoder(bar_batch)   # (H, n_summary, D)
    for slot in range(H):
        s = slot * n_summary
        memory[0, s:s+n_summary]       = embs[slot]
        mem_key_mask[0, s:s+n_summary] = False

    return memory, mem_key_mask
