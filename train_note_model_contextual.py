"""
Retrain ONLY the note model with cross-bar conditioning.

What changed vs the original training:
  - tokenizer.encode_bar now includes up to MAX_PREV_BAR (48) tokens from the
    previous bar in the prefix, giving the model actual cross-bar memory.
  - make_note_model() uses seq_len=512 (up from 256) so dense bars still fit.
  - NoteDataset._transpose also transposes pitch tokens in the prefix.

Run on Lightning AI (or any GPU box):
    python preprocess.py                         # rebuild note_samples.pkl
    python train_note_model_contextual.py        # train note model only

Upload to Lightning AI:
    - All .py files in this directory
    - data/ directory (MIDI files)
    - checkpoints/chord_model.pt (keep existing chord model, only note model retrains)

After training, download checkpoints/note_model.pt and replace the local copy.
"""

import argparse, math, os, pickle, time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import NoteDataset
from models  import make_note_model


def cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, optimizer, scaler, scheduler, device, grad_clip=1.0):
    model.train()
    total_loss, n = 0.0, 0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad(set_to_none=True)
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
def eval_epoch(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            _, loss = model(x, y, loss_mask=mask)
        total_loss += loss.item()
        n          += 1
    return total_loss / max(n, 1)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--batch_size',  type=int,   default=128)
    parser.add_argument('--lr',          type=float, default=4e-4)
    parser.add_argument('--val_split',   type=float, default=0.05)
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--device',      default='auto')
    parser.add_argument('--save_path',   default='checkpoints/note_model.pt')
    args = parser.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
        if args.device == 'auto' else args.device
    )
    print(f"Device: {device}")

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    print("Loading note samples…")
    with open('data/processed/note_samples.pkl', 'rb') as f:
        note_samples = pickle.load(f)
    print(f"  {len(note_samples)} bar samples")

    # Log prefix length distribution so we can confirm the new format
    prefix_lens = [len(p) for p, _ in note_samples[:1000]]
    avg_prefix  = sum(prefix_lens) / len(prefix_lens)
    print(f"  Avg prefix length (first 1000): {avg_prefix:.1f} tokens")
    print(f"  (Should be ~30+ if prev-bar context is included; 4 = old format)")

    n_val   = max(1, int(len(note_samples) * args.val_split))
    n_train = len(note_samples) - n_val
    rng     = torch.Generator().manual_seed(42)
    train_idx, val_idx = random_split(range(len(note_samples)), [n_train, n_val], generator=rng)
    train_samples = [note_samples[i] for i in train_idx]
    val_samples   = [note_samples[i] for i in val_idx]

    # seq_len=512 matches make_note_model() — samples that exceed this are dropped
    train_ds = NoteDataset(train_samples, seq_len=320, augment=True,  max_shift=5)
    val_ds   = NoteDataset(val_samples,   seq_len=320, augment=False)
    print(f"  Train samples after seq_len filter: {len(train_ds)}")
    print(f"  Val   samples after seq_len filter: {len(val_ds)}")

    kw_shared = dict(num_workers=args.num_workers, pin_memory=True,
                     persistent_workers=(args.num_workers > 0), prefetch_factor=4)
    train_l = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                         drop_last=True, **kw_shared)
    val_l   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **kw_shared)

    model = make_note_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nNote model: {n_params:,} params  seq_len={model.seq_len}")

    cfg = model_config(model)

    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = args.epochs * len(train_l)
    warmup_steps = min(1000, total_steps // 10)
    scheduler    = cosine_schedule(optimizer, warmup_steps, total_steps)
    scaler       = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    print(f"Epochs: {args.epochs}  |  warmup steps: {warmup_steps}\n")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0       = time.time()
        tr_loss  = run_epoch(model, train_l, optimizer, scaler, scheduler, device)
        val_loss = eval_epoch(model, val_l, device)
        elapsed  = time.time() - t0

        marker = ''
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'config': cfg, 'state': model.state_dict()},
                       args.save_path)
            marker = '  ← saved'

        print(f"  [{epoch:3d}/{args.epochs}] "
              f"train={tr_loss:.4f}  val={val_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s{marker}")

    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Saved to: {args.save_path}")


if __name__ == '__main__':
    main()
