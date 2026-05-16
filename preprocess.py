"""
Run once to tokenize all MIDI files.
Saves two pickle files to data/processed/:
  chord_sequences.pkl  — list of token lists (chord model training data)
  note_samples.pkl     — list of (prefix, note_tokens) tuples (note model training data)

Usage:
    python preprocess.py
"""

import os, glob, pickle
from tqdm import tqdm
from tokenizer import midi_to_training_data, CHORD_VOCAB, NOTE_VOCAB, QUALITIES, ROOTS


def main():
    midi_files = sorted(
        glob.glob('data/**/*.mid',  recursive=True) +
        glob.glob('data/**/*.midi', recursive=True)
    )
    print(f"Found {len(midi_files)} MIDI files\n")

    chord_sequences = []
    note_samples    = []
    failed = skipped = 0

    for path in tqdm(midi_files, desc="Tokenising"):
        chord_seq, bar_samples = midi_to_training_data(path)
        if chord_seq is None:
            failed += 1
            continue
        if len(chord_seq) <= 4:   # BOS + 1 bar pair + EOS = too short
            skipped += 1
            continue
        chord_sequences.append(chord_seq)
        if bar_samples:
            note_samples.extend(bar_samples)

    print(f"\nResults:")
    print(f"  Parsed OK   : {len(chord_sequences)} files")
    print(f"  Failed      : {failed}")
    print(f"  Too short   : {skipped}")
    print(f"  Chord seqs  : {len(chord_sequences)}")
    print(f"  Bar samples : {len(note_samples)}")

    # Quick vocab sanity check
    flat = [t for seq in chord_sequences for t in seq]
    print(f"\nChord token range: {min(flat)}–{max(flat)}  (vocab size {CHORD_VOCAB})")
    flat_notes = [t for prefix, notes in note_samples for t in prefix + notes]
    print(f"Note  token range: {min(flat_notes)}–{max(flat_notes)}  (vocab size {NOTE_VOCAB})")

    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/chord_sequences.pkl', 'wb') as f:
        pickle.dump(chord_sequences, f)
    with open('data/processed/note_samples.pkl', 'wb') as f:
        pickle.dump(note_samples, f)

    print("\nSaved to data/processed/  — ready to train.")


if __name__ == '__main__':
    main()
