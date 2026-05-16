"""
Train the chord model and/or the note model.

Usage:
    python train.py                          # train both (default)
    python train.py --stage chords           # chord model only
    python train.py --stage notes            # note model only
    python train.py --stage both --chord_epochs 150 --note_epochs 80

Checkpoints are saved to checkpoints/ whenever validation loss improves.
"""

import argparse, math, os, pickle, time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import ChordDataset, NoteDataset
from models  import make_chord_model, make_note_model


# ──────────────────────────────────────────────
#  LR schedule: linear warmup + cosine decay
# ──────────────────────────────────────────────
def cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────
#  One training epoch
# ──────────────────────────────────────────────
def run_epoch(model, loader, optimizer, scaler, scheduler, device,
              is_note_model, grad_clip=1.0):
    model.train()
    total_loss, n = 0.0, 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)

        if is_note_model:
            x, y, mask = (t.to(device) for t in batch)
        else:
            x, y = (t.to(device) for t in batch)
            mask = None

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            _, loss = model(x, y, loss_mask=mask)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        n          += 1

    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, is_note_model):
    model.eval()
    total_loss, n = 0.0, 0
    for batch in loader:
        if is_note_model:
            x, y, mask = (t.to(device) for t in batch)
        else:
            x, y = (t.to(device) for t in batch)
            mask = None
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            _, loss = model(x, y, loss_mask=mask)
        total_loss += loss.item()
        n          += 1
    return total_loss / max(n, 1)


# ──────────────────────────────────────────────
#  Full training loop for one model
# ──────────────────────────────────────────────
def train(model, train_loader, val_loader, epochs, lr, save_path,
          name, device, is_note_model):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps   = epochs * len(train_loader)
    warmup_steps  = min(1000, total_steps // 10)
    scheduler     = cosine_schedule(optimizer, warmup_steps, total_steps)
    scaler        = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n── {name} model  ({n_params:,} params) ──")
    print(f"   train batches: {len(train_loader):,}  |  val batches: {len(val_loader):,}")
    print(f"   epochs: {epochs}  |  warmup steps: {warmup_steps}\n")

    best_val, best_epoch = float('inf'), 0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0       = time.time()
        tr_loss  = run_epoch(model, train_loader, optimizer, scaler,
                             scheduler, device, is_note_model)
        val_loss = eval_epoch(model, val_loader, device, is_note_model)
        elapsed  = time.time() - t0

        marker = ''
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            torch.save({'config': model_config(model), 'state': model.state_dict()},
                       save_path)
            marker = '  ← saved'

        print(f"  [{epoch:3d}/{epochs}] "
              f"train={tr_loss:.4f}  val={val_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s{marker}")

    print(f"\n  Best val loss {best_val:.4f} at epoch {best_epoch}")


def model_config(model):
    emb   = model.tok_emb
    block = model.blocks[0]
    return dict(
        vocab_size = emb.num_embeddings,
        d_model    = emb.embedding_dim,
        n_heads    = block.attn.num_heads,
        n_layers   = len(model.blocks),
        seq_len    = model.seq_len,
        dropout    = block.ff[2].p,
    )


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',        choices=['chords','notes','both'], default='both')
    parser.add_argument('--chord_epochs', type=int,   default=120)
    parser.add_argument('--note_epochs',  type=int,   default=80)
    parser.add_argument('--batch_size',   type=int,   default=64)
    parser.add_argument('--lr',           type=float, default=3e-4)
    parser.add_argument('--val_split',    type=float, default=0.05,
                        help='Fraction of data held out for validation')
    parser.add_argument('--num_workers',  type=int,   default=2)
    parser.add_argument('--device',       default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    def split(dataset, val_frac):
        n_val   = max(1, int(len(dataset) * val_frac))
        n_train = len(dataset) - n_val
        return random_split(dataset, [n_train, n_val],
                            generator=torch.Generator().manual_seed(42))

    def make_loaders(dataset, val_frac, batch_size, workers, collate_fn=None):
        train_ds, val_ds = split(dataset, val_frac)
        kw = dict(batch_size=batch_size, num_workers=workers, pin_memory=True)
        if collate_fn:
            kw['collate_fn'] = collate_fn
        return (DataLoader(train_ds, shuffle=True,  **kw),
                DataLoader(val_ds,   shuffle=False, **kw))

    # ── Stage 1: chord model ──────────────────
    if args.stage in ('chords', 'both'):
        print("\nLoading chord sequences…")
        with open('data/processed/chord_sequences.pkl', 'rb') as f:
            chord_seqs = pickle.load(f)
        print(f"  {len(chord_seqs)} sequences")

        ds            = ChordDataset(chord_seqs, seq_len=256)
        train_l, val_l = make_loaders(ds, args.val_split, args.batch_size, args.num_workers)

        model = make_chord_model().to(device)
        train(model, train_l, val_l,
              epochs=args.chord_epochs, lr=args.lr,
              save_path='checkpoints/chord_model.pt',
              name='Chord', device=device, is_note_model=False)

    # ── Stage 2: note model ───────────────────
    if args.stage in ('notes', 'both'):
        print("\nLoading note samples…")
        with open('data/processed/note_samples.pkl', 'rb') as f:
            note_samples = pickle.load(f)
        print(f"  {len(note_samples)} bar samples")

        import math as _math
        n_val   = max(1, int(len(note_samples) * args.val_split))
        n_train = len(note_samples) - n_val
        rng     = torch.Generator().manual_seed(42)
        train_idx, val_idx = torch.utils.data.random_split(
            range(len(note_samples)), [n_train, n_val], generator=rng)
        train_samples = [note_samples[i] for i in train_idx]
        val_samples   = [note_samples[i] for i in val_idx]

        train_ds = NoteDataset(train_samples, seq_len=256, augment=True,  max_shift=5)
        val_ds   = NoteDataset(val_samples,   seq_len=256, augment=False)
        kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
        train_l = DataLoader(train_ds, shuffle=True,  **kw)
        val_l   = DataLoader(val_ds,   shuffle=False, **kw)

        model = make_note_model().to(device)
        train(model, train_l, val_l,
              epochs=args.note_epochs, lr=args.lr,
              save_path='checkpoints/note_model.pt',
              name='Note', device=device, is_note_model=True)

    print("\nDone.")


if __name__ == '__main__':
    main()
