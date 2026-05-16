"""
ElegyBox — Hierarchical Note Model Training
============================================
Run this on Lightning AI (T4 GPU).  Two-phase training:

  Phase 1  (~15 epochs) : freeze the existing note model, train only the
                          BarEncoder and the per-block cross-attention adapters.
  Phase 2  (~20 epochs) : unfreeze everything and fine-tune jointly at a
                          very small learning rate so the note model learns
                          to use the cross-attention signal.

Architecture additions (no changes to existing files):
  BarEncoder              — small bidirectional transformer (d_model=128, 3 layers)
                            compresses each completed bar into N_SUMMARY=4 vectors
                            then projects them to 384-dim for cross-attention.
  _BlockWithCrossAttn     — drop-in replacement for _Block; adds one cross-attention
                            sub-layer after self-attention. Existing weights load via
                            strict=False (ln1/ln2/attn/ff keys are identical).
  HierarchicalMusicGPT    — MusicGPT with _BlockWithCrossAttn blocks. Accepts an
                            optional `memory` tensor from the bar encoder.

Key transposition augmentation:  ±6 semitones, applied consistently across
the history bars, prefix, and note tokens for each training sample.

After training completes, update generate.py to:
  1. Load BarEncoder + HierarchicalMusicGPT checkpoints.
  2. After generating each bar, encode it and append to the running memory.
  3. Pass memory into generate_bar (cross-attention is a no-op when memory=None,
     so bar 0 works unchanged).
"""

# %% ── 0. Environment ────────────────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "mido", "--quiet"],
               check=True)

import os, math, random, time, pickle
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ── Paths — adjust if your Lightning AI layout differs ──────────────────────
REPO_DIR  = Path('/teamspace/studios/this_studio/ElegyBox')
DATA_DIR  = REPO_DIR / 'data'
CKPT_DIR  = REPO_DIR / 'checkpoints'
CKPT_DIR.mkdir(exist_ok=True)

# Existing epoch-50 checkpoint (upload this before running)
NOTE_CKPT  = CKPT_DIR / 'note_model.pt'
HIER_CKPT  = CKPT_DIR / 'note_model_hierarchical.pt'   # output
ENC_CKPT   = CKPT_DIR / 'bar_encoder.pt'               # output

# Upload this from your local machine (run preprocess_hierarchical.py first)
HIER_DATA  = DATA_DIR / 'processed' / 'note_samples_hierarchical.pkl'

sys.path.insert(0, str(REPO_DIR))
from tokenizer import (
    NOTE_PAD, NOTE_BAR_END,
    NOTE_ROOT_OFF, NOTE_QUAL_OFF, NOTE_NONE,
    NOTE_ON_OFF, NOTE_DUR_OFF, NOTE_VEL_OFF, NOTE_VOCAB,
    midi_to_training_data,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")
if device.type == 'cuda':
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# %% ── 1. Config ─────────────────────────────────────────────────────────────
# ── Bar encoder ──────────────────────────────────────────────────────────────
N_SUMMARY    = 4      # summary vectors produced per bar
ENC_D_MODEL  = 128    # bar encoder internal width
ENC_N_HEADS  = 4
ENC_N_LAYERS = 3
ENC_MAX_LEN  = 200    # max bar tokens fed to the encoder (longer bars are truncated)
NOTE_D_MODEL = 384    # note model d_model — bar encoder projects to this

# ── History window ────────────────────────────────────────────────────────────
MAX_HISTORY  = 16     # max bars of encoder context (bars 0..N-2 for bar N)
              #        16 bars × 4 summary tokens = 64 memory tokens per step

# ── Augmentation ─────────────────────────────────────────────────────────────
MAX_SHIFT    = 6      # key transposition ±6 semitones

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 32
NUM_WORKERS  = 2
VAL_SPLIT    = 0.05
GRAD_CLIP    = 1.0

PHASE1_EPOCHS = 15
PHASE1_LR     = 3e-4   # adapters + encoder (random init → need higher lr)

PHASE2_EPOCHS = 20
PHASE2_LR_NEW = 1e-4   # adapters + encoder (continued fine-tune)
PHASE2_LR_OLD = 5e-6   # existing note model weights (very conservative)

SEQ_LEN = 320           # unchanged from existing note model

_NOTE_ON_END = NOTE_ON_OFF + 88   # pitches 21-108  →  tokens 41-128
_N_ROOTS     = 12


# %% ── 2. Architecture ───────────────────────────────────────────────────────

# ── 2a. BarEncoder ───────────────────────────────────────────────────────────

class _BidirBlock(nn.Module):
    """Non-causal transformer block used inside the bar encoder."""
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
        # Full (bidirectional) attention — bar is complete, no causal mask needed.
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


class BarEncoder(nn.Module):
    """
    Compresses a completed bar's note tokens into N_SUMMARY fixed vectors.

    Prepends N_SUMMARY learnable "summary" tokens to the bar sequence, runs
    bidirectional attention, and returns the first N_SUMMARY output vectors
    projected to the note model's d_model (BERT-style CLS tokens).

    Parameters:  ~1.5 M  (fast to train, small memory footprint)
    """
    def __init__(self, vocab_size=NOTE_VOCAB, d_enc=ENC_D_MODEL,
                 n_heads=ENC_N_HEADS, n_layers=ENC_N_LAYERS,
                 max_len=ENC_MAX_LEN, n_summary=N_SUMMARY,
                 note_d_model=NOTE_D_MODEL):
        super().__init__()
        self.n_summary = n_summary
        self.max_len   = max_len
        self.tok_emb   = nn.Embedding(vocab_size, d_enc)
        self.pos_emb   = nn.Embedding(max_len + n_summary, d_enc)
        # Learnable summary tokens prepended before bar tokens
        self.summary   = nn.Parameter(torch.zeros(1, n_summary, d_enc))
        nn.init.normal_(self.summary, std=0.02)
        self.blocks    = nn.ModuleList(
            [_BidirBlock(d_enc, n_heads) for _ in range(n_layers)]
        )
        self.ln_f      = nn.LayerNorm(d_enc)
        # Project from encoder width to note model width for cross-attention
        self.proj      = nn.Linear(d_enc, note_d_model)

    def forward(self, bar_tokens):
        """
        bar_tokens : (B, T)  — token ids, 0-padded if needed
        Returns    : (B, n_summary, note_d_model)
        """
        B, T   = bar_tokens.shape
        T      = min(T, self.max_len)
        toks   = bar_tokens[:, :T]

        x      = self.tok_emb(toks)                          # (B, T, d_enc)
        summ   = self.summary.expand(B, -1, -1)              # (B, n_summary, d_enc)
        x      = torch.cat([summ, x], dim=1)                 # (B, n_summary+T, d_enc)
        pos    = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x      = x + self.pos_emb(pos)

        for blk in self.blocks:
            x  = blk(x)
        x      = self.ln_f(x)
        return self.proj(x[:, :self.n_summary])              # (B, n_summary, note_d_model)


# ── 2b. HierarchicalMusicGPT ─────────────────────────────────────────────────

class _BlockWithCrossAttn(nn.Module):
    """
    Drop-in for models._Block.  Identical self-attention + FF sub-layers
    (same parameter names → existing weights load via strict=False), plus a
    cross-attention sub-layer to attend to bar-encoder memory.
    When memory=None the block behaves identically to the original _Block.
    """
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
        # Cross-attention adapter (new weights — not in existing checkpoint)
        self.ln_cross   = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)

    def forward(self, x, causal_mask, memory=None, memory_key_mask=None):
        # ── Self-attention (unchanged from original _Block) ──────────────
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + h
        # ── Cross-attention to bar history (skipped on bar 0) ────────────
        if memory is not None:
            h = self.ln_cross(x)
            h, _ = self.cross_attn(h, memory, memory,
                                   key_padding_mask=memory_key_mask,
                                   need_weights=False)
            x = x + h
        # ── Feed-forward (unchanged) ─────────────────────────────────────
        x = x + self.ff(self.ln2(x))
        return x


class HierarchicalMusicGPT(nn.Module):
    """
    MusicGPT with cross-attention adapters.

    Identical to models.MusicGPT in all self-attention and FF parameters, so
    an existing note_model.pt checkpoint loads cleanly with strict=False.
    The cross-attention weights (ln_cross, cross_attn) are new and start
    randomly initialised.
    """
    def __init__(self, vocab_size, d_model, n_heads, n_layers, seq_len,
                 dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList(
            [_BlockWithCrossAttn(d_model, n_heads, dropout)
             for _ in range(n_layers)]
        )
        self.ln_f    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight          # weight tying

        causal = torch.triu(torch.full((seq_len, seq_len), float('-inf')),
                            diagonal=1)
        self.register_buffer('causal_mask', causal, persistent=False)

    def forward(self, x, targets=None, loss_mask=None,
                memory=None, memory_key_mask=None):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        h    = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        mask = self.causal_mask[:T, :T]

        for blk in self.blocks:
            h = blk(h, mask, memory=memory,
                    memory_key_mask=memory_key_mask)
        h      = self.ln_f(h)
        logits = self.head(h)

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=0, reduction='none',
        )
        if loss_mask is not None:
            denom = loss_mask.reshape(-1).sum().clamp(min=1)
            loss  = (loss * loss_mask.reshape(-1)).sum() / denom
        else:
            valid = (targets != 0).reshape(-1)
            loss  = loss[valid].mean()
        return logits, loss


def make_hierarchical_note_model():
    return HierarchicalMusicGPT(
        vocab_size=153, d_model=384, n_heads=8, n_layers=8,
        seq_len=320, dropout=0.1,
    )


# %% ── 3. Data ────────────────────────────────────────────────────────────────

# ── 3a. Token-level augmentation (used by the dataset) ───────────────────────

def _transpose_token_list(tokens, semitones):
    """
    Transpose a list of note-vocab tokens by `semitones`.
    Shifts NOTE_ON pitch tokens and chord ROOT tokens.
    Quality, position, duration, velocity tokens are unchanged.
    """
    if semitones == 0:
        return tokens
    result = []
    for tok in tokens:
        if NOTE_ON_OFF <= tok < _NOTE_ON_END:
            new_idx = max(0, min(_NOTE_ON_END - NOTE_ON_OFF - 1,
                                 (tok - NOTE_ON_OFF) + semitones))
            result.append(NOTE_ON_OFF + new_idx)
        elif NOTE_ROOT_OFF <= tok < NOTE_ROOT_OFF + _N_ROOTS:
            result.append(NOTE_ROOT_OFF + (tok - NOTE_ROOT_OFF + semitones) % _N_ROOTS)
        else:
            result.append(tok)
    return result


# Load pre-processed hierarchical dataset.
# Run  preprocess_hierarchical.py  on your LOCAL machine first, then upload
# data/processed/note_samples_hierarchical.pkl to Lightning AI.
if not HIER_DATA.exists():
    raise FileNotFoundError(
        f"\n\n  Missing: {HIER_DATA}\n\n"
        "  Generate it locally with:\n"
        "      python preprocess_hierarchical.py\n\n"
        "  then upload  data/processed/note_samples_hierarchical.pkl  here."
    )

print(f"Loading hierarchical data from {HIER_DATA}")
with open(HIER_DATA, 'rb') as f:
    all_samples = pickle.load(f)
print(f"  {len(all_samples):,} samples loaded")


# ── 3b. Dataset ───────────────────────────────────────────────────────────────

class HierarchicalNoteDataset(Dataset):
    """
    Wraps the hierarchical samples list.
    Each item:
        x          (T,)   input token ids
        targets    (T,)   next-token targets
        loss_mask  (T,)   1.0 where loss is computed (note tokens only)
        enc_inputs list[list[int]]  history bar token lists for bar encoder
    """
    def __init__(self, samples, seq_len=SEQ_LEN, augment=True,
                 max_shift=MAX_SHIFT, max_history=MAX_HISTORY):
        self.seq_len     = seq_len
        self.augment     = augment
        self.max_shift   = max_shift
        self.max_history = max_history
        # Filter out samples too long to fit in seq_len
        self.samples = [
            s for s in samples
            if len(s[1]) + len(s[2]) <= seq_len + 1
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        history, prefix, notes = self.samples[idx]

        # Deep-copy mutable lists before augmenting
        history = [list(h) for h in history]
        prefix  = list(prefix)
        notes   = list(notes)

        # Key transposition: shift applied consistently across all token lists
        if self.augment and self.max_shift > 0:
            shift = random.randint(-self.max_shift, self.max_shift)
            if shift != 0:
                history = [_transpose_token_list(h, shift) for h in history]
                prefix  = _transpose_token_list(prefix, shift)
                notes   = _transpose_token_list(notes,  shift)

        # Trim history to window
        history = history[-self.max_history:]

        # Build model input sequence
        full = prefix + notes
        need = self.seq_len + 1
        if len(full) < need:
            full = full + [NOTE_PAD] * (need - len(full))
        full = full[:need]

        # Loss mask: 1 starting at position that predicts the first note token
        mask       = [0.0] * len(full)
        note_start = len(prefix) - 1     # x[note_start] → predicts notes[0]
        for j in range(note_start, len(mask)):
            mask[j] = 1.0

        x       = torch.tensor(full[:-1], dtype=torch.long)
        targets = torch.tensor(full[1:],  dtype=torch.long)
        lmask   = torch.tensor(mask[1:],  dtype=torch.float)
        lmask[targets == NOTE_PAD] = 0.0   # never train on PAD targets

        return x, targets, lmask, history   # history is a list-of-lists


def hierarchical_collate(batch):
    """
    Pad x / targets / loss_mask to the longest sequence in the batch.
    history is returned as a list-of-lists (processed per-sample in the loop).
    """
    xs, targets, masks, histories = zip(*batch)

    max_len = max(x.size(0) for x in xs)
    B       = len(xs)

    x_pad = torch.zeros(B, max_len, dtype=torch.long)
    t_pad = torch.zeros(B, max_len, dtype=torch.long)
    m_pad = torch.zeros(B, max_len, dtype=torch.float)

    for i, (x, t, m) in enumerate(zip(xs, targets, masks)):
        L = x.size(0)
        x_pad[i, :L] = x
        t_pad[i, :L] = t
        m_pad[i, :L] = m

    return x_pad, t_pad, m_pad, list(histories)


# ── Train / val split ─────────────────────────────────────────────────────────
random.seed(42)
shuffled = list(all_samples)
random.shuffle(shuffled)

n_val   = max(1, int(len(shuffled) * VAL_SPLIT))
val_s   = shuffled[:n_val]
train_s = shuffled[n_val:]

train_ds = HierarchicalNoteDataset(train_s, augment=True)
val_ds   = HierarchicalNoteDataset(val_s,   augment=False)

print(f"Train: {len(train_ds):,}   Val: {len(val_ds):,}")

loader_kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                 pin_memory=True, collate_fn=hierarchical_collate)
train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)


# %% ── 4. Training Utilities ──────────────────────────────────────────────────

def encode_histories(bar_encoder, histories, device,
                     n_summary=N_SUMMARY, max_history=MAX_HISTORY):
    """
    Encode all history bars across the batch in a single bar-encoder forward pass.

    histories : list of B elements, each a list of token lists (variable length).

    Returns
    -------
    memory         (B, max_history*n_summary, note_d_model)
    memory_key_mask (B, max_history*n_summary)  — True where padding (ignore)
    """
    B   = len(histories)
    D   = bar_encoder.proj.out_features
    ML  = max_history * n_summary

    memory       = torch.zeros(B, ML, D, device=device)
    mem_key_mask = torch.ones(B, ML, dtype=torch.bool, device=device)  # True = ignore

    # Collect all (batch_idx, slot, tokens) items into one GPU batch
    items = []
    for b, bars in enumerate(histories):
        bars = bars[-max_history:]
        for slot, bar_toks in enumerate(bars):
            items.append((b, slot, bar_toks))

    if not items:
        return memory, mem_key_mask

    # Pad all bars to same length and stack
    max_bar_len = min(max(len(it[2]) for it in items), bar_encoder.max_len)
    bar_batch   = torch.zeros(len(items), max_bar_len, dtype=torch.long,
                              device=device)
    for i, (_, _, toks) in enumerate(items):
        t_trunc = toks[:max_bar_len]
        bar_batch[i, :len(t_trunc)] = torch.tensor(t_trunc, dtype=torch.long,
                                                    device=device)

    embs = bar_encoder(bar_batch)   # (N_items, n_summary, D)

    for i, (b, slot, _) in enumerate(items):
        s = slot * n_summary
        e = s + n_summary
        memory[b, s:e]       = embs[i]
        mem_key_mask[b, s:e] = False   # valid positions

    return memory, mem_key_mask


def cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * t)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(note_model, bar_encoder, loader, optimizer, scaler,
              scheduler, device, train=True):
    note_model.train(train)
    bar_encoder.train(train)
    total_loss, n = 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, targets, loss_mask, histories in loader:
            x         = x.to(device)
            targets   = targets.to(device)
            loss_mask = loss_mask.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                memory, mem_key_mask = encode_histories(
                    bar_encoder, histories, device)
                _, loss = note_model(x, targets, loss_mask,
                                     memory=memory,
                                     memory_key_mask=mem_key_mask)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    list(note_model.parameters()) +
                    list(bar_encoder.parameters()),
                    GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            total_loss += loss.item()
            n          += 1

    return total_loss / max(n, 1)


def count_params(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(note_model, bar_encoder, epoch, val_loss):
    torch.save({
        'epoch':    epoch,
        'val_loss': val_loss,
        'config': dict(vocab_size=153, d_model=384, n_heads=8, n_layers=8,
                       seq_len=320, dropout=0.1),
        'state':  note_model.state_dict(),
    }, HIER_CKPT)
    torch.save({
        'epoch':    epoch,
        'val_loss': val_loss,
        'config': dict(vocab_size=NOTE_VOCAB, d_enc=ENC_D_MODEL,
                       n_heads=ENC_N_HEADS, n_layers=ENC_N_LAYERS,
                       max_len=ENC_MAX_LEN, n_summary=N_SUMMARY,
                       note_d_model=NOTE_D_MODEL),
        'state':  bar_encoder.state_dict(),
    }, ENC_CKPT)
    print(f"  ✓ Saved  epoch={epoch}  val={val_loss:.4f}")


# %% ── 5. Instantiate models & load existing weights ─────────────────────────

bar_encoder  = BarEncoder().to(device)
note_model   = make_hierarchical_note_model().to(device)

# Load the epoch-50 note model weights into the hierarchical model.
# strict=False: existing keys (ln1/ln2/attn/ff) load perfectly;
# new cross-attention keys (ln_cross/cross_attn) are left randomly init'd.
assert NOTE_CKPT.exists(), f"Checkpoint not found: {NOTE_CKPT}"
ckpt = torch.load(NOTE_CKPT, map_location=device, weights_only=False)
missing, unexpected = note_model.load_state_dict(ckpt['state'], strict=False)
print(f"\nLoaded note model from {NOTE_CKPT}")
print(f"  Missing  (new cross-attn): {len(missing)} keys")
print(f"  Unexpected (should be 0) : {len(unexpected)} keys")

new_keys = [k for k in missing if 'cross_attn' in k or 'ln_cross' in k]
assert len(new_keys) == len(missing), \
    f"Unexpected missing keys (not cross-attn): {set(missing) - set(new_keys)}"
print(f"  All {len(missing)} missing keys are cross-attention adapter weights ✓")


# %% ── 6. Phase 1 — train adapters + bar encoder only ────────────────────────
# Freeze everything in the note model except the new cross-attention layers.

def set_cross_attn_requires_grad(model, requires_grad):
    for name, param in model.named_parameters():
        if 'cross_attn' in name or 'ln_cross' in name:
            param.requires_grad = requires_grad
        else:
            param.requires_grad = False if not requires_grad else param.requires_grad

# Phase 1: only cross-attn adapters + bar encoder are trainable
for param in note_model.parameters():
    param.requires_grad = False
for name, param in note_model.named_parameters():
    if 'cross_attn' in name or 'ln_cross' in name:
        param.requires_grad = True

print(f"\n── Phase 1 ─────────────────────────────────────────────────────")
print(f"   Trainable params — note model adapters : "
      f"{count_params(note_model):>10,}")
print(f"   Trainable params — bar encoder         : "
      f"{count_params(bar_encoder):>10,}")

p1_params   = ([p for p in note_model.parameters() if p.requires_grad]
               + list(bar_encoder.parameters()))
optimizer   = torch.optim.AdamW(p1_params, lr=PHASE1_LR, weight_decay=0.01)
total_steps = PHASE1_EPOCHS * len(train_loader)
warmup      = min(500, total_steps // 10)
scheduler   = cosine_schedule(optimizer, warmup, total_steps)
scaler      = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

best_val = float('inf')
print(f"   Epochs: {PHASE1_EPOCHS}   LR: {PHASE1_LR}   "
      f"Batches/epoch: {len(train_loader):,}\n")

for epoch in range(1, PHASE1_EPOCHS + 1):
    t0      = time.time()
    tr_loss = run_epoch(note_model, bar_encoder, train_loader,
                        optimizer, scaler, scheduler, device, train=True)
    vl_loss = run_epoch(note_model, bar_encoder, val_loader,
                        optimizer, scaler, scheduler, device, train=False)
    elapsed = time.time() - t0

    marker = ''
    if vl_loss < best_val:
        best_val = vl_loss
        save_checkpoint(note_model, bar_encoder, epoch, vl_loss)
        marker = '  ← saved'

    print(f"  P1 [{epoch:2d}/{PHASE1_EPOCHS}]  "
          f"train={tr_loss:.4f}  val={vl_loss:.4f}  "
          f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.0f}s{marker}")

print(f"\nPhase 1 complete.  Best val: {best_val:.4f}")


# %% ── 7. Phase 2 — unfreeze all, fine-tune jointly ──────────────────────────
# Two parameter groups: existing note model weights get a very small lr
# to preserve what was learned; adapters + encoder get a moderate lr.

print(f"\n── Phase 2 ─────────────────────────────────────────────────────")

# Unfreeze entire note model
for param in note_model.parameters():
    param.requires_grad = True

old_params = [p for name, p in note_model.named_parameters()
              if 'cross_attn' not in name and 'ln_cross' not in name]
new_params = ([p for name, p in note_model.named_parameters()
               if 'cross_attn' in name or 'ln_cross' in name]
              + list(bar_encoder.parameters()))

print(f"   Old note model params (lr={PHASE2_LR_OLD}) : "
      f"{sum(p.numel() for p in old_params):>10,}")
print(f"   New adapter+encoder  (lr={PHASE2_LR_NEW}) : "
      f"{sum(p.numel() for p in new_params):>10,}")

optimizer = torch.optim.AdamW(
    [{'params': old_params, 'lr': PHASE2_LR_OLD},
     {'params': new_params, 'lr': PHASE2_LR_NEW}],
    weight_decay=0.01,
)
total_steps = PHASE2_EPOCHS * len(train_loader)
warmup      = min(300, total_steps // 10)
scheduler   = cosine_schedule(optimizer, warmup, total_steps)
scaler      = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

print(f"   Epochs: {PHASE2_EPOCHS}   Batches/epoch: {len(train_loader):,}\n")

for epoch in range(1, PHASE2_EPOCHS + 1):
    t0      = time.time()
    tr_loss = run_epoch(note_model, bar_encoder, train_loader,
                        optimizer, scaler, scheduler, device, train=True)
    vl_loss = run_epoch(note_model, bar_encoder, val_loader,
                        optimizer, scaler, scheduler, device, train=False)
    elapsed = time.time() - t0

    marker = ''
    if vl_loss < best_val:
        best_val = vl_loss
        save_checkpoint(note_model, bar_encoder, epoch, vl_loss)
        marker = '  ← saved'

    print(f"  P2 [{epoch:2d}/{PHASE2_EPOCHS}]  "
          f"train={tr_loss:.4f}  val={vl_loss:.4f}  "
          f"lr_old={optimizer.param_groups[0]['lr']:.1e}  "
          f"lr_new={optimizer.param_groups[1]['lr']:.1e}  "
          f"{elapsed:.0f}s{marker}")

print(f"\nPhase 2 complete.  Best val overall: {best_val:.4f}")
print(f"Checkpoints saved to:\n  {HIER_CKPT}\n  {ENC_CKPT}")


# %% ── 8. What changes in generate.py (inference) ────────────────────────────
"""
After training, update generate.py as follows:

1.  Add to imports:
        from models_hierarchical import HierarchicalMusicGPT, BarEncoder

2.  load_model() already works — HierarchicalMusicGPT has the same config
    keys.  Load bar encoder separately:

        note_model = HierarchicalMusicGPT(**ckpt['config']).to(device)
        note_model.load_state_dict(ckpt['state'])

        enc_ckpt   = torch.load('checkpoints/bar_encoder.pt', ...)
        bar_encoder = BarEncoder(**enc_ckpt['config']).to(device)
        bar_encoder.load_state_dict(enc_ckpt['state'])
        bar_encoder.eval()

3.  Add running bar history list before the generation loop:
        bar_history = []   # list of raw note-token lists

4.  In the per-bar loop, before generate_bar():
        # Encode history into memory for this bar
        if bar_history:
            memory, mem_key_mask = encode_histories(
                bar_encoder, [bar_history], device)
        else:
            memory = mem_key_mask = None

5.  Pass memory into generate_bar() (update the signature to accept and
    forward memory + mem_key_mask to note_model()).

6.  After generate_bar() returns raw_tokens:
        bar_history.append(raw_tokens)
        if section != prev_bar_section:
            bar_history = []   # reset at section boundaries (same logic as prev_bar_tokens)

That's it — the chord model is unchanged.
"""
print("\nDone.  See Section 8 comments for the generate.py changes needed.")
