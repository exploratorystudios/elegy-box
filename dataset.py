"""
PyTorch datasets for both training stages.
"""

import random
import torch
from torch.utils.data import Dataset

# Note-vocab constants (must match tokenizer.py)
_NOTE_ROOT_OFF = 2
_NOTE_ON_OFF   = 41
_NOTE_DUR_OFF  = 129   # first token after pitch range
_N_ROOTS       = 12
_N_PITCHES     = 88    # pitches 21-108


class ChordDataset(Dataset):
    """
    One sample = one chord sequence (BOS … EOS), padded / chunked to seq_len+1.
    Returns (x, y) where y = x shifted left by 1.
    """

    def __init__(self, sequences, seq_len=256):
        self.seq_len = seq_len
        self.samples = []

        for seq in sequences:
            if len(seq) <= seq_len + 1:
                self.samples.append(list(seq))
            else:
                # Very long pieces: sliding window with 50 % stride
                stride = seq_len // 2
                for start in range(0, len(seq) - seq_len, stride):
                    self.samples.append(seq[start: start + seq_len + 1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq  = self.samples[idx]
        need = self.seq_len + 1
        if len(seq) < need:
            seq = seq + [0] * (need - len(seq))
        seq  = seq[:need]
        x    = torch.tensor(seq[:-1], dtype=torch.long)
        y    = torch.tensor(seq[1:],  dtype=torch.long)
        return x, y


class NoteDataset(Dataset):
    """
    One sample = one bar: prefix (4 chord tokens) + note_tokens.

    x, y  : standard autoregressive shift
    mask  : float, 1 where loss is computed (note tokens only, not prefix)

    The prefix is given — the model learns to predict note tokens conditioned on it.
    Loss is computed starting from the position that first *predicts* a note token
    (i.e. position prefix_len - 1 in x, whose target is note_tokens[0]).
    """

    def __init__(self, samples, seq_len=256, augment=True, max_shift=5):
        self.seq_len   = seq_len
        self.augment   = augment
        self.max_shift = max_shift
        self.samples   = []

        for prefix, notes in samples:
            total = len(prefix) + len(notes)
            if total <= seq_len + 1:
                self.samples.append((list(prefix), list(notes)))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _transpose(prefix, notes, shift):
        """Shift all chord roots and pitch-ON tokens by `shift` semitones.

        The prefix now contains prev-bar NOTE_ON tokens in addition to chord
        root tokens, so both token ranges must be transposed consistently.
        """
        new_prefix = list(prefix)
        for i, tok in enumerate(new_prefix):
            if _NOTE_ROOT_OFF <= tok < _NOTE_ROOT_OFF + _N_ROOTS:
                # Chord root token — rotate within the 12 roots
                new_prefix[i] = _NOTE_ROOT_OFF + (tok - _NOTE_ROOT_OFF + shift) % _N_ROOTS
            elif _NOTE_ON_OFF <= tok < _NOTE_DUR_OFF:
                # Pitch token from the prev-bar context — shift and clamp
                new_pitch = (tok - _NOTE_ON_OFF) + shift
                new_pitch = max(0, min(_N_PITCHES - 1, new_pitch))
                new_prefix[i] = _NOTE_ON_OFF + new_pitch

        new_notes = list(notes)
        for i, tok in enumerate(new_notes):
            if _NOTE_ON_OFF <= tok < _NOTE_DUR_OFF:
                new_pitch = (tok - _NOTE_ON_OFF) + shift
                new_pitch = max(0, min(_N_PITCHES - 1, new_pitch))
                new_notes[i] = _NOTE_ON_OFF + new_pitch

        return new_prefix, new_notes

    def __getitem__(self, idx):
        prefix, notes = self.samples[idx]

        if self.augment and self.max_shift > 0:
            shift = random.randint(-self.max_shift, self.max_shift)
            if shift != 0:
                prefix, notes = self._transpose(prefix, notes, shift)

        full = prefix + notes

        need = self.seq_len + 1
        if len(full) < need:
            full = full + [0] * (need - len(full))
        full = full[:need]

        x = torch.tensor(full[:-1], dtype=torch.long)
        y = torch.tensor(full[1:],  dtype=torch.long)

        # Mask: 1 starting at the position that predicts the first note token
        mask = torch.zeros(len(x), dtype=torch.float)
        start = len(prefix) - 1          # x[start] = last prefix token → predicts notes[0]
        mask[start:] = 1.0
        mask[y == 0] = 0.0               # never train on PAD targets

        return x, y, mask
