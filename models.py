"""
Shared GPT architecture for both the chord-sequence model and the note-filling model.

Both are small decoder-only transformers (pre-LN, weight-tied head).
The note model uses a loss_mask to ignore the chord-conditioning prefix tokens.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, causal_mask):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


class MusicGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, seq_len, dropout=0.1):
        super().__init__()
        self.seq_len  = seq_len
        self.tok_emb  = nn.Embedding(vocab_size, d_model)
        self.pos_emb  = nn.Embedding(seq_len, d_model)
        self.drop     = nn.Dropout(dropout)
        self.blocks   = nn.ModuleList(
            [_Block(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f     = nn.LayerNorm(d_model)
        self.head     = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight   # weight tying

        causal = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', causal, persistent=False)

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------
    def forward(self, x, targets=None, loss_mask=None):
        """
        x           : (B, T)  token ids
        targets     : (B, T)  shifted targets, optional
        loss_mask   : (B, T)  float mask – 1 where loss is computed, 0 elsewhere
        """
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        h    = self.drop(self.tok_emb(x) + self.pos_emb(pos))

        mask = self.causal_mask[:T, :T]
        for block in self.blocks:
            h = block(h, mask)

        h      = self.ln_f(h)
        logits = self.head(h)

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=0,          # PAD
            reduction='none',
        )
        if loss_mask is not None:
            denom = loss_mask.reshape(-1).sum().clamp(min=1)
            loss  = (loss * loss_mask.reshape(-1)).sum() / denom
        else:
            valid = (targets != 0).reshape(-1)
            loss  = loss[valid].mean()

        return logits, loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, temperature=1.0,
                 top_k=40, top_p=0.9, stop_token=None, forbidden=None):
        """
        prompt : (1, T) seed tokens already on the correct device
        Returns (1, T + generated) tensor.
        """
        self.eval()
        x = prompt.clone()

        for _ in range(max_new_tokens):
            ctx     = x[:, -self.seq_len:]
            logits  = self(ctx)[:, -1, :].float()   # (1, V)

            if forbidden:
                for t in forbidden:
                    logits[:, t] = float('-inf')

            logits  = logits / max(temperature, 1e-8)

            if top_k > 0:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < vals[:, [-1]]] = float('-inf')

            if top_p < 1.0:
                sorted_l, sorted_i = torch.sort(logits, descending=True)
                cum_p = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
                remove = cum_p - F.softmax(sorted_l, dim=-1) > top_p
                sorted_l[remove] = float('-inf')
                logits = torch.zeros_like(logits).scatter_(1, sorted_i, sorted_l)

            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            x        = torch.cat([x, next_tok], dim=1)

            if stop_token is not None and next_tok.item() == stop_token:
                break

        return x


# ------------------------------------------------------------------
#  Named constructors – keeps hyperparameters in one place
# ------------------------------------------------------------------
def make_chord_model():
    return MusicGPT(
        vocab_size=26,
        d_model=128, n_heads=4, n_layers=4, seq_len=256, dropout=0.1,
    )

def make_note_model():
    # seq_len=320: prefix is at most 52 tokens (4 chord + 48 prev-bar notes from
    # MAX_PREV_BAR), leaving 268 tokens for note generation — sufficient for all
    # but the densest bars. Reduced from 512 to cut O(T²) attention cost by 2.5x.
    return MusicGPT(
        vocab_size=153,
        d_model=384, n_heads=8, n_layers=8, seq_len=320, dropout=0.1,
    )
