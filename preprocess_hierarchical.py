"""
preprocess_hierarchical.py — run this LOCALLY before uploading to Lightning AI.

Reads all MIDI files in data/, builds the hierarchical training samples
(history, prefix, notes) for each bar, and saves the result as:

    data/processed/note_samples_hierarchical.pkl

Upload that pkl to Lightning AI alongside the rest of the repo. The training
notebook (train_hierarchical.py) will detect it and skip the build step.

Usage:
    python preprocess_hierarchical.py
    python preprocess_hierarchical.py --data_dir /path/to/midi/root
    python preprocess_hierarchical.py --max_history 16
"""

import argparse
import glob
import pickle
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))

from tokenizer import (
    NOTE_PAD, NOTE_BAR_END,
    NOTE_ROOT_OFF, NOTE_QUAL_OFF, NOTE_NONE,
    NOTE_ON_OFF, NOTE_DUR_OFF, NOTE_VEL_OFF, NOTE_VOCAB,
    midi_to_training_data,
)

_NOTE_ON_END = NOTE_ON_OFF + 88   # pitches 21-108  →  tokens 41-128
_N_ROOTS     = 12


def build_hierarchical_samples(midi_glob, max_history=16, save_path=None):
    """
    For each bar N in every MIDI file, produce:
        history  : list of note-token lists for bars 0..N-2
        prefix   : chord-conditioning prefix for bar N
        notes    : note tokens for bar N  (training target)

    The immediately preceding bar (N-1) is already embedded in the prefix via
    prev_bar_tokens, so the encoder history starts at N-2 and reaches back
    max_history bars further.
    """
    midi_files = sorted(glob.glob(midi_glob, recursive=True))
    print(f"Found {len(midi_files)} MIDI files")

    samples = []
    failed  = 0

    for path in midi_files:
        try:
            _, bar_samples = midi_to_training_data(path)
        except Exception as e:
            print(f"  SKIP {Path(path).name}: {e}")
            failed += 1
            continue
        if not bar_samples:
            continue

        for i, (prefix, notes) in enumerate(bar_samples):
            start   = max(0, i - 1 - max_history)
            end     = max(0, i - 1)
            history = [bar_samples[j][1] for j in range(start, end)]
            samples.append((history, list(prefix), list(notes)))

    print(f"Done — {len(samples):,} samples built, {failed} files skipped")

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'wb') as f:
            pickle.dump(samples, f)
        size_mb = out.stat().st_size / 1e6
        print(f"Saved → {out}  ({size_mb:.1f} MB)")

    return samples


def main():
    parser = argparse.ArgumentParser(description='Build hierarchical training data locally.')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Root directory to search for .mid files (default: data/)')
    parser.add_argument('--max_history', type=int, default=16,
                        help='Max bars of encoder context per sample (default: 16)')
    parser.add_argument('--output', type=str,
                        default='data/processed/note_samples_hierarchical.pkl',
                        help='Output pkl path')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data directory not found: {data_dir.resolve()}")
        sys.exit(1)

    midi_glob = str(data_dir / '**' / '*.mid')
    build_hierarchical_samples(midi_glob, max_history=args.max_history,
                               save_path=args.output)
    print()
    print("Next steps:")
    print(f"  1. Upload {args.output} to Lightning AI alongside the repo.")
    print("  2. Run train_hierarchical.py — it will detect the pkl and skip rebuilding.")


if __name__ == '__main__':
    main()
