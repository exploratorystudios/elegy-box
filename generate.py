"""
Generate a classical MIDI using the two trained models.

Grammar-constrained decoding is used for both models so every generated
token is structurally valid — no silent drops, no dead bars.

Usage:
    python generate.py                            # 32 bars, 90 bpm → generated.mid
    python generate.py --n_bars 64 --bpm 72 --output nocturne.mid
    python generate.py --chord_temp 0.8 --note_temp 1.0 --top_p 0.85
"""

import argparse
import math
import random
import mido
import torch
import torch.nn.functional as F

from tokenizer import (
    # chord vocab
    CHORD_BOS, CHORD_EOS, CHORD_PAD,
    CHORD_ROOT_OFF, CHORD_QUAL_OFF, CHORD_NONE, CHORD_VOCAB,
    # note vocab
    NOTE_PAD, NOTE_BAR_END,
    NOTE_ROOT_OFF, NOTE_QUAL_OFF, NOTE_NONE,
    NOTE_POS_OFF, NOTE_ON_OFF, NOTE_DUR_OFF, NOTE_VEL_OFF, NOTE_VOCAB,
    POSITIONS, MAX_DUR, N_VEL_BINS, MAX_PREV_BAR,
    # helpers
    ROOTS, QUALITIES,
    decode_chord_sequence, decode_bar_notes, bars_to_midi,
    midi_to_training_data,
)
from models import MusicGPT
from models_hierarchical import (
    load_hierarchical_note_model, load_bar_encoder, build_inference_memory,
)


# ── Token range constants (derived from tokenizer) ───────────────────
_NOTE_POS_END = NOTE_POS_OFF + POSITIONS           # 41
_NOTE_ON_END  = NOTE_ON_OFF  + (109 - 21)          # 129  (pitches 21-108)
_NOTE_DUR_END = NOTE_DUR_OFF + MAX_DUR             # 145
_NOTE_VEL_END = NOTE_VEL_OFF + N_VEL_BINS          # 153

_CHORD_ROOT_END = CHORD_ROOT_OFF + 12              # 15
_CHORD_QUAL_END = CHORD_QUAL_OFF + 10              # 25


# ── Load helper ───────────────────────────────────────────────────────
def load_model(path, device):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = MusicGPT(**ckpt['config']).to(device)
    # causal_mask is a non-persistent buffer (computed in __init__); strip it
    # from any checkpoint that saved it before persistent=False was set
    state = {k: v for k, v in ckpt['state'].items() if k != 'causal_mask'}
    model.load_state_dict(state)
    model.eval()
    return model


def _load_note_model(device):
    """
    Load the note model.  Prefers the hierarchical checkpoint when available;
    falls back to the original.  HierarchicalMusicGPT with memory=None is a
    perfect drop-in for MusicGPT, so inference is identical before training.
    """
    import os
    hier_path = 'checkpoints/note_model_hierarchical.pt'
    base_path = 'checkpoints/note_model.pt'
    if os.path.exists(hier_path):
        print("Loading hierarchical note model…")
        return load_hierarchical_note_model(hier_path, device)
    print("Loading note model (upgrading to hierarchical architecture)…")
    return load_hierarchical_note_model(base_path, device)


def _load_bar_encoder(device):
    import os
    path = 'checkpoints/bar_encoder.pt'
    if os.path.exists(path):
        print("Loading bar encoder…")
        return load_bar_encoder(path, device)
    print("No bar_encoder.pt found — cross-attention disabled (memory=None).")
    return None


# ── Sampling kernel ───────────────────────────────────────────────────
def _sample(logits, temperature, top_k, top_p):
    """Apply temperature + top-k + top-p to a (1, V) logits tensor."""
    logits = logits.float() / max(temperature, 1e-8)

    if top_k > 0:
        finite = (logits > float('-inf')).sum().item()
        k = min(top_k, max(1, finite))
        vals, _ = torch.topk(logits, k)
        logits[logits < vals[:, [-1]]] = float('-inf')

    if top_p < 1.0:
        sorted_l, sorted_i = torch.sort(logits, descending=True)
        cum_p  = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
        remove = cum_p - F.softmax(sorted_l, dim=-1) > top_p
        sorted_l[remove] = float('-inf')
        logits = torch.zeros_like(logits).scatter_(1, sorted_i, sorted_l)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)


# ── Chord generation (grammar-constrained) ────────────────────────────
# Chord token stream: BOS [ROOT QUAL]* EOS
# Alternating states: expect ROOT (or EOS) vs expect QUAL

_MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]  # W W H W W W H

def diatonic_roots_for_key(key_root):
    """Return the set of 7 diatonic root indices for the given key root (0=C…11=B)."""
    return {(key_root + interval) % 12 for interval in _MAJOR_INTERVALS}


_MAJOR_KEY_QUALITIES = {
    0:  {0, 5},      # I: maj / maj7
    2:  {1, 6},      # ii: min / min7
    4:  {1, 6},      # iii: min / min7
    5:  {0, 5},      # IV: maj / maj7
    7:  {0, 4},      # V: maj / dom7
    9:  {1, 6},      # vi: min / min7
    11: {2, 7},      # vii: dim / hdim7
}


def _major_key_quality_mask(root, key_root, strength):
    """Soft logit penalties for qualities that do not fit the major-key scale degree."""
    mask = torch.zeros(CHORD_VOCAB)
    if key_root is None or root is None:
        return mask
    degree = (root - key_root) % 12
    allowed = _MAJOR_KEY_QUALITIES.get(degree)
    if not allowed:
        return mask
    for q in range(len(QUALITIES)):
        if q not in allowed:
            mask[CHORD_QUAL_OFF + q] -= strength
    return mask


def _scaled_strength(value, scale=0.5):
    return value * scale


def generate_chords(model, n_bars, temperature, top_k, top_p, device, max_repeat=3,
                    diatonic_roots=None, chord_key_strength=3.0, key_root=None):
    """
    Grammar-constrained chord generation with sliding-window repetition penalty.

    FSM states:
      EXPECT_ROOT       → ROOT token, CHORD_NONE, or CHORD_EOS
      EXPECT_QUAL_REAL  → QUAL token only  (previous ROOT was a real pitch class)
      EXPECT_QUAL_NONE  → CHORD_NONE only  (previous ROOT was CHORD_NONE)

    Penalty: we look back over a window of 2*max_repeat completed bars and apply
    a logit penalty of PENALTY_PER_HIT per occurrence of each root.  With a wide
    enough window a penalised chord stays suppressed long enough that the model
    can't immediately bounce back to it after one "escape" bar.
    """
    from collections import Counter

    EXPECT_ROOT      = 0
    EXPECT_QUAL_REAL = 1
    EXPECT_QUAL_NONE = 2

    WINDOW          = max_repeat * 4   # look-back length (wider window keeps hard block active longer)
    PENALTY_PER_HIT = 3.5              # logit subtracted per occurrence in window

    # recent: list of root indices (int) or None, one entry per completed bar
    recent   = []
    cur_root = None
    state    = EXPECT_ROOT

    def chord_mask():
        mask = torch.full((CHORD_VOCAB,), float('-inf'))
        if state == EXPECT_ROOT:
            mask[CHORD_ROOT_OFF:_CHORD_ROOT_END] = 0.0
            mask[CHORD_NONE] = 0.0
            if len(recent) >= n_bars:   # suppress EOS until we have enough bars
                mask[CHORD_EOS] = 0.0
            # Diatonic key bias: penalise non-diatonic roots so the piece stays in key
            if diatonic_roots is not None:
                for r in range(12):
                    if r not in diatonic_roots:
                        if mask[CHORD_ROOT_OFF + r] > float('-inf'):
                            mask[CHORD_ROOT_OFF + r] -= _scaled_strength(chord_key_strength)
            # Sliding-window frequency penalty
            counts = Counter(recent[-WINDOW:])
            for item, cnt in counts.items():
                if item is None:
                    # Hard block CHORD_NONE after 2 occurrences in window
                    if cnt >= 2:
                        mask[CHORD_NONE] = float('-inf')
                    else:
                        mask[CHORD_NONE] -= PENALTY_PER_HIT
                else:
                    tok = CHORD_ROOT_OFF + item
                    if mask[tok] > float('-inf'):
                        if cnt >= max_repeat:
                            mask[tok] = float('-inf')   # hard block
                        else:
                            mask[tok] -= PENALTY_PER_HIT * cnt
        elif state == EXPECT_QUAL_REAL:
            mask[CHORD_QUAL_OFF:_CHORD_QUAL_END] = 0.0
            if key_root is not None and cur_root is not None:
                root_idx = cur_root - CHORD_ROOT_OFF
                mask += _major_key_quality_mask(root_idx, key_root, _scaled_strength(chord_key_strength))
        else:  # EXPECT_QUAL_NONE
            mask[CHORD_NONE] = 0.0
        return mask

    x = torch.tensor([[CHORD_BOS]], device=device)

    for _ in range(n_bars * 2 + 2):
        ctx    = x[:, -model.seq_len:]
        logits = model(ctx)[:, -1, :]
        logits = logits + chord_mask().to(device)
        tok    = _sample(logits, temperature, top_k, top_p)
        val    = tok.item()
        x      = torch.cat([x, tok], dim=1)

        if val == CHORD_EOS:
            break

        if state == EXPECT_ROOT:
            cur_root = val
            state    = EXPECT_QUAL_NONE if val == CHORD_NONE else EXPECT_QUAL_REAL
        else:
            # Completed a bar — record root in history
            root_idx = None if cur_root == CHORD_NONE else (cur_root - CHORD_ROOT_OFF)
            recent.append(root_idx)
            state = EXPECT_ROOT

    return decode_chord_sequence(x[0].tolist())


def _postprocess_chords(chords, max_run=3):
    """
    Clean up two failure modes that slip through generation:
      1. Consecutive None runs > 1 bar  (EOS-padding or attractor collapse)
      2. Consecutive identical chord runs > max_run bars
    Applied after generation so the fixes are guaranteed regardless of how the
    model terminated.
    """
    if not chords:
        return chords
    result = list(chords)
    n = len(result)

    # Pass 0 — fix leading None: piece shouldn't start without a chord
    first_real = next((c for c in result if c is not None), None)
    if result[0] is None and first_real is not None:
        result[0] = first_real

    # Pass 1 — collapse None runs: keep the first None in each run, replace the rest
    last_real = next((c for c in result if c is not None), None)
    prev_was_none = False
    for i in range(n):
        if result[i] is None:
            if prev_was_none and last_real is not None:
                result[i] = last_real
            prev_was_none = True
        else:
            last_real   = result[i]
            prev_was_none = False

    # Pass 2 — break long repetition runs: when a chord has repeated max_run
    # times in a row, replace this bar by cycling through all distinct chords
    # seen in the wider context (so we don't just oscillate between two chords).
    cycle_idx = {}   # stuck_chord → index into its candidate list
    for i in range(1, n):
        if result[i] is None:
            continue
        run = 1
        j   = i - 1
        while j >= 0 and result[j] == result[i]:
            run += 1
            j   -= 1
        if run > max_run:
            stuck = result[i]
            # Collect ordered-unique alternatives from a wide lookback window
            seen, candidates = set(), []
            for k in range(max(0, i - max_run * 4), i):
                c = result[k]
                if c != stuck and c is not None and c not in seen:
                    candidates.append(c)
                    seen.add(c)
            if candidates:
                idx         = cycle_idx.get(stuck, 0) % len(candidates)
                result[i]   = candidates[idx]
                cycle_idx[stuck] = idx + 1

    # Pass 3 — smooth jarring root jumps: if a chord has an "unstable" quality
    # (dim / hdim7 / aug) AND its root is ≥ 5 semitones from the previous chord's root,
    # replace it with the previous chord for one bar (hold instead of jump).
    # This prevents chromatic outlier chords (e.g. E dim in a Bb-minor context) from
    # feeling like a left-turn, while leaving intentional tense chords (like a dom7
    # building into a cadence) untouched.
    TENSE_QUALITIES = {2, 3, 7}  # dim, aug, hdim7
    for i in range(1, n):
        if result[i] is None or result[i - 1] is None:
            continue
        r0, q0 = result[i - 1]
        r1, q1 = result[i]
        if q1 not in TENSE_QUALITIES:
            continue
        root_dist = min((r1 - r0) % 12, (r0 - r1) % 12)
        if root_dist >= 5:
            result[i] = result[i - 1]

    return result


# ── Note generation (grammar-constrained) ─────────────────────────────
# Note token stream (after chord prefix):
#   (POSITION (NOTE_ON DURATION VELOCITY)*)* BAR_END
#
# 3-state automaton:
#   FREE      → valid next: POSITION, NOTE_ON, BAR_END
#   NEED_DUR  → valid next: DURATION only
#   NEED_VEL  → valid next: VELOCITY only

FREE     = 0
NEED_DUR = 1
NEED_VEL = 2

_NOTE_MASKS = {}   # cached per (state, bar_end_allowed, pos_allowed, device)

def _note_mask(state, device, bar_end_ok=True, pos_ok=True):
    key = (state, bar_end_ok, pos_ok, device)
    if key not in _NOTE_MASKS:
        mask = torch.full((NOTE_VOCAB,), float('-inf'), device=device)
        if state == FREE:
            if bar_end_ok:
                mask[NOTE_BAR_END] = 0.0
            if pos_ok:
                mask[NOTE_POS_OFF:_NOTE_POS_END] = 0.0
            mask[NOTE_ON_OFF :_NOTE_ON_END ] = 0.0
        elif state == NEED_DUR:
            mask[NOTE_DUR_OFF:_NOTE_DUR_END] = 0.0
        elif state == NEED_VEL:
            mask[NOTE_VEL_OFF:_NOTE_VEL_END] = 0.0
        _NOTE_MASKS[key] = mask
    return _NOTE_MASKS[key]


def _next_note_state(state, tok):
    if state == FREE:
        return NEED_DUR if NOTE_ON_OFF <= tok < _NOTE_ON_END else FREE
    if state == NEED_DUR:
        return NEED_VEL
    return FREE   # NEED_VEL → FREE


def build_metric_bias(beat_strength, offbeat_penalty, device):
    """
    Logit bias on position tokens to create metric hierarchy.
      pos 0, 8  (beats 1, 3) → +beat_strength
      pos 4, 12 (beats 2, 4) → +beat_strength * 0.4
      pos 2,6,10,14 (8th off-beats) → 0
      remaining 16th off-beats   → offbeat_penalty
    """
    bias = torch.zeros(NOTE_VOCAB, device=device)
    for pos in range(POSITIONS):
        tok = NOTE_POS_OFF + pos
        if pos in (0, 8):
            bias[tok] = beat_strength
        elif pos in (4, 12):
            bias[tok] = beat_strength * 0.4
        elif pos % 2 == 0:          # 8th-note positions (2,6,10,14) — neutral
            bias[tok] = 0.0
        else:                        # pure 16th off-beats (1,3,5,7,9,11,13,15)
            bias[tok] = offbeat_penalty
    return bias


def generate_bar(model, prefix, temperature, top_k, top_p, device,
                 max_tokens=192, min_notes=4, min_chord_notes=2,
                 min_positions=4, max_chord_notes=5,
                 extra_bias=None, beat_bass_bias=None,
                 position_gap_penalty=0.3,
                 pitch_repeat_penalty=1.5,
                 dur_repeat_penalty=0.8,
                 memory=None, memory_key_mask=None):
    """
    Generate note tokens for one bar with grammar-constrained decoding.

    min_notes             : BAR_END suppressed until this many total notes placed.
    min_chord_notes       : min notes per rhythmic position (forces dyads/triads).
    min_positions         : min distinct rhythmic positions before BAR_END allowed.
                            Prevents bars that close after 1-2 slots.
    max_chord_notes       : hard cap on notes per position; once hit, ON tokens are
                            masked so the model must advance to the next position.
                            Prevents the bias from piling 10+ notes into one slot.
    extra_bias            : optional (1, NOTE_VOCAB) logit bias added every step.
    beat_bass_bias        : optional (NOTE_VOCAB,) pitch bias applied on NOTE_ON
                            tokens only when the current position is a strong beat
                            (pos 0 or 8). Steers the first note of each strong beat
                            toward the bass register.
    position_gap_penalty  : logit penalty per skipped position slot when choosing
                            the next POSITION token. Penalises jumping from pos N
                            to pos N+k by k * penalty, reducing unnatural long rests
                            within a bar without forbidding shorter rests.
    """
    x                = torch.tensor([prefix], device=device)
    state            = FREE
    notes_placed     = 0
    positions_filled = 0
    # Training bars always begin with a POSITION token, and positions are emitted
    # in ascending order. Preserve that grammar at generation time; otherwise the
    # model is asked to continue sequences it never saw during training.
    notes_at_cur_pos = min_chord_notes
    cur_pos          = -1                # last position token chosen (0-15)
    cur_pos_pitches  = set()            # pitch indices already placed at cur_pos
    recent_pitches   = []               # list of sets, newest first, max depth 3
    recent_durations = []              # duration indices used recently, newest first, max depth 3
    last_note_on     = None             # NOTE_ON pitch index of in-progress note

    for _ in range(max_tokens):
        ctx    = x[:, -model.seq_len:]
        logits = model(ctx, memory=memory, memory_key_mask=memory_key_mask)[:, -1, :]

        is_beat_pos     = cur_pos in (0, 4, 8, 12)
        position_floor  = min_chord_notes if (positions_filled <= 1 or is_beat_pos) else 1
        chord_satisfied = notes_at_cur_pos >= position_floor
        slot_full       = notes_at_cur_pos >= max_chord_notes
        bar_end_ok = chord_satisfied \
                     and (positions_filled >= min_positions) \
                     and (notes_placed >= min_notes) \
                     and (cur_pos >= 8)
        next_pos   = cur_pos + 1
        pos_ok     = chord_satisfied and next_pos < POSITIONS
        on_ok      = (cur_pos >= 0) and not slot_full

        # Build mask — extend _note_mask to handle on_ok
        mask = torch.full((NOTE_VOCAB,), float('-inf'), device=device)
        if state == FREE:
            if bar_end_ok:
                mask[NOTE_BAR_END] = 0.0
            if pos_ok:
                mask[NOTE_POS_OFF + next_pos:_NOTE_POS_END] = 0.0
                if position_gap_penalty > 0:
                    for gap in range(1, POSITIONS - next_pos):
                        tok = NOTE_POS_OFF + next_pos + gap
                        if tok < _NOTE_POS_END:
                            mask[tok] -= position_gap_penalty * gap
            if on_ok:
                mask[NOTE_ON_OFF:_NOTE_ON_END] = 0.0
                # Hard-block pitches already placed at this position — no chord unisons
                for pi in cur_pos_pitches:
                    mask[NOTE_ON_OFF + pi] = float('-inf')
                # Penalise pitches from recent positions with decay so the model
                # can't cycle A→B→A even within a restricted chord-tone set.
                if pitch_repeat_penalty > 0:
                    for age, past in enumerate(recent_pitches):
                        decay = 1.0 / (age + 1)
                        for pi in past - cur_pos_pitches:
                            idx = NOTE_ON_OFF + pi
                            if mask[idx] != float('-inf'):
                                mask[idx] -= pitch_repeat_penalty * decay
        elif state == NEED_DUR:
            mask[NOTE_DUR_OFF:_NOTE_DUR_END] = 0.0
            if dur_repeat_penalty > 0:
                for age, dur_idx in enumerate(recent_durations):
                    decay = 1.0 / (age + 1)
                    mask[NOTE_DUR_OFF + dur_idx] -= dur_repeat_penalty * decay
        elif state == NEED_VEL:
            mask[NOTE_VEL_OFF:_NOTE_VEL_END] = 0.0

        if state == FREE and not torch.isfinite(mask).any():
            # If monotonic position constraints leave no legal continuation,
            # close the bar instead of feeding all -inf logits into sampling.
            mask[NOTE_BAR_END] = 0.0

        logits = logits + mask
        if extra_bias is not None:
            logits = logits + extra_bias
        # Bass register pull on strong beats: applied only when choosing a NOTE_ON
        # in FREE state on position 0 or 8 (beats 1 and 3).
        if (beat_bass_bias is not None and state == FREE
                and cur_pos in (0, 8) and on_ok):
            logits = logits + beat_bass_bias

        tok  = _sample(logits, temperature, top_k, top_p)
        val  = tok.item()
        x    = torch.cat([x, tok], dim=1)

        prev_state = state
        state = _next_note_state(state, val)

        if prev_state == FREE and NOTE_POS_OFF <= val < _NOTE_POS_END:
            recent_pitches = ([cur_pos_pitches] + recent_pitches)[:3]
            cur_pos        = val - NOTE_POS_OFF
            notes_at_cur_pos  = 0
            positions_filled += 1
            cur_pos_pitches   = set()
        if prev_state == FREE and NOTE_ON_OFF <= val < _NOTE_ON_END:
            last_note_on = val - NOTE_ON_OFF
        if prev_state == NEED_DUR and NOTE_DUR_OFF <= val < _NOTE_DUR_END:
            dur_idx = val - NOTE_DUR_OFF
            recent_durations = ([dur_idx] + recent_durations)[:3]
        if prev_state == NEED_VEL:
            notes_placed     += 1
            notes_at_cur_pos += 1
            if last_note_on is not None:
                cur_pos_pitches.add(last_note_on)
                last_note_on = None

        if val == NOTE_BAR_END:
            break

    note_tokens = x[0, len(prefix):].tolist()
    # Return both decoded events (for post-processing) and raw tokens (for
    # the next bar's prefix so the model has cross-bar memory).
    return decode_bar_notes(note_tokens), note_tokens


# ── Chord → note-model prefix ─────────────────────────────────────────
def chord_prefix(prev, curr, prev_bar_tokens=None):
    """Build the prefix tensor for generate_bar.

    Matches the tokenizer.encode_bar prefix format:
        [prev_root, prev_qual, <prev_bar_tokens[-MAX_PREV_BAR:]>, curr_root, curr_qual]

    prev_bar_tokens: raw note tokens from the previous bar (the generate_bar
    return value).  Pass None for the first bar or after a section boundary.
    """
    def enc(c):
        return [NOTE_NONE, NOTE_NONE] if c is None else \
               [NOTE_ROOT_OFF + c[0], NOTE_QUAL_OFF + c[1]]
    prev_ctx = list(prev_bar_tokens[-MAX_PREV_BAR:]) if prev_bar_tokens else []
    return enc(prev) + prev_ctx + enc(curr)


# ── Coherence helpers ─────────────────────────────────────────────────

_MIN_PITCH = 21   # matches tokenizer MIN_PITCH

def build_key_bias(chords, device, strength=0.7, tonic_root=None, mode=None):
    """
    Return a (NOTE_VOCAB,) logit bias that boosts pitches in the active key.
    If tonic_root is provided, use it directly; otherwise infer a fallback from
    the chord progression.
    Returns None if chords are too ambiguous to infer a key.
    """
    from collections import Counter
    MAJOR_INTERVALS = {0, 2, 4, 5, 7, 9, 11}
    MINOR_INTERVALS = {0, 2, 3, 5, 7, 8, 10}

    qual_counts = Counter(c[1] for c in chords if c is not None)
    if tonic_root is None:
        root_counts = Counter(c[0] for c in chords if c is not None)
        if not root_counts:
            return None
        tonic_root = root_counts.most_common(1)[0][0]

    if mode is None:
        mode = 'minor' if qual_counts.get(1, 0) > qual_counts.get(0, 0) else 'major'
    intervals = MINOR_INTERVALS if mode == 'minor' else MAJOR_INTERVALS
    in_key = {(tonic_root + iv) % 12 for iv in intervals}

    bias = torch.zeros(NOTE_VOCAB, device=device)
    for tok in range(NOTE_ON_OFF, _NOTE_ON_END):
        pitch_class = (tok - NOTE_ON_OFF + _MIN_PITCH) % 12
        if pitch_class in in_key:
            bias[tok] = strength
    return bias


def build_chord_tone_bias(chord, device, strength=0.75):
    """
    Per-bar logit bias that boosts NOTE_ON tokens whose pitch class belongs to
    the current bar's chord (root, 3rd, 5th, 7th).  Applied on top of the
    global key bias so chord tones get a double boost and diatonic non-chord
    tones are merely in-key — creating a clear harmonic hierarchy:
        chord tones  > diatonic passing tones > chromatic notes
    """
    if chord is None:
        return None
    root, qual = chord
    intervals = _QUALITY_INTERVALS[qual] if qual < len(_QUALITY_INTERVALS) else [0, 4, 7]
    chord_pcs  = {(root + iv) % 12 for iv in intervals}

    bias = torch.zeros(NOTE_VOCAB, device=device)
    for tok in range(NOTE_ON_OFF, _NOTE_ON_END):
        pc = (tok - NOTE_ON_OFF + _MIN_PITCH) % 12
        if pc in chord_pcs:
            bias[tok] = strength
    return bias


def _resolve_none_chords(chords, tonic_root=None):
    """
    Replace generated None chords before note generation.

    None is common in the training labels, but downstream generation uses the
    current chord for key filtering and procedural bass. Letting None through
    creates bars with no harmonic anchor and often no left hand at all.
    """
    fallback = (tonic_root, 0) if tonic_root is not None else next(
        (c for c in chords if c is not None), (0, 0))
    result = []
    last_real = fallback
    for chord in chords:
        if chord is None:
            result.append(last_real)
        else:
            result.append(chord)
            last_real = chord
    return result


def register_bias(register_center, sigma, strength, device):
    """
    Gaussian logit bias over pitch-ON tokens centred on register_center (MIDI pitch).
    Pulls generation toward the previous bar's average pitch.
    """
    pitches = torch.arange(NOTE_ON_OFF, _NOTE_ON_END, device=device).float()
    midi    = pitches - NOTE_ON_OFF + _MIN_PITCH
    bias    = torch.zeros(NOTE_VOCAB, device=device)
    bias[NOTE_ON_OFF:_NOTE_ON_END] = strength * torch.exp(
        -0.5 * ((midi - register_center) / sigma) ** 2
    )
    return bias


def build_dur_bias(strength, device):
    """
    Log-ramp logit bias over duration tokens — boosts longer durations so the
    model sustains notes rather than generating staccato blips.

    Duration tokens: NOTE_DUR_OFF + i for i in 0..(MAX_DUR-1)
    representing 1..MAX_DUR 1/16th-note durations.

    The bias is 0 at dur=1 and `strength` at dur=MAX_DUR, on a log scale so
    the push toward medium durations (4-8/16) is strong without forcing
    everything to be a half-note.
    """
    import math
    bias = torch.zeros(NOTE_VOCAB, device=device)
    log_max = math.log(MAX_DUR + 1)
    for i in range(MAX_DUR):
        bias[NOTE_DUR_OFF + i] = strength * math.log(i + 2) / log_max
    return bias


# ── Form helpers ─────────────────────────────────────────────────────

def _distribute(total, n_parts):
    """Distribute total items into n_parts as evenly as possible."""
    base, extra = divmod(total, n_parts)
    return [base + (1 if i < extra else 0) for i in range(n_parts)]


def _apply_form(chords, n_bars, form):
    """
    Reorder chords according to form and return (new_chords, section_labels).
    section_labels[i] is 'A', 'B', or 'C' for each bar.
    """
    n = n_bars

    if form == 'aba' and n >= 6:
        a = n // 3; b = n - 2 * a
        new_chords = chords[:a] + chords[a:a+b] + chords[:a]
        labels = ['A']*a + ['B']*b + ['A']*a
        return new_chords[:n], labels

    elif form == 'binary' and n >= 4:
        h = n // 2
        return chords[:n], ['A']*h + ['B']*(n - h)

    elif form == 'rondo' and n >= 7:
        # ABACABA: 7 sections — A recurs, B and C are episodes
        seg_lens = _distribute(n, 7)
        pool_A_len = seg_lens[0] + seg_lens[2] + seg_lens[4] + seg_lens[6]
        pool_B_len = seg_lens[1] + seg_lens[5]
        pool_C_len = seg_lens[3]
        pool_A = chords[:pool_A_len] or chords[:1]
        pool_B = chords[pool_A_len:pool_A_len+pool_B_len] or chords[:1]
        pool_C = chords[pool_A_len+pool_B_len:pool_A_len+pool_B_len+pool_C_len] or chords[:1]
        pattern = ['A','B','A','C','A','B','A']
        pools   = {'A': pool_A, 'B': pool_B, 'C': pool_C}
        pos     = {'A': 0, 'B': 0, 'C': 0}
        new_chords, labels = [], []
        for lbl, cnt in zip(pattern, seg_lens):
            pool = pools[lbl]
            for j in range(cnt):
                new_chords.append(pool[(pos[lbl] + j) % len(pool)])
                labels.append(lbl)
            pos[lbl] += cnt
        return new_chords[:n], labels[:n]

    elif form == 'arch' and n >= 5:
        # ABCBA: 5 segments — builds to C then mirrors back
        seg_lens = _distribute(n, 5)
        s = [sum(seg_lens[:i]) for i in range(6)]
        seg_a = chords[s[0]:s[1]]
        seg_b = chords[s[1]:s[2]]
        seg_c = chords[s[2]:s[3]]
        new_chords = seg_a + seg_b + seg_c + list(reversed(seg_b)) + list(reversed(seg_a))
        labels = (['A']*seg_lens[0] + ['B']*seg_lens[1] + ['C']*seg_lens[2]
                  + ['B']*seg_lens[3] + ['A']*seg_lens[4])
        return new_chords[:n], labels[:n]

    elif form == 'variation' and n >= 3:
        # Three repetitions of the same chord progression, generation params escalate
        seg_lens = _distribute(n, 3)
        base = chords[:seg_lens[0]] or chords[:1]
        new_chords, labels = [], []
        for var_lbl, seg_len in zip(['A','B','C'], seg_lens):
            seg = base * ((seg_len // len(base)) + 1)
            new_chords.extend(seg[:seg_len])
            labels.extend([var_lbl] * seg_len)
        return new_chords[:n], labels[:n]

    # 'none' or unsupported
    return chords[:n], ['A'] * n


def _apply_phrase_repetition(bar_events_list, section_labels, phrase_bars, repeat_prob):
    """
    Post-processing pass: ABAB period structure.
    Within each section, collect phrases then walk in groups of 4.
    With probability repeat_prob, copy phrases 0,1 over phrases 2,3
    within each 4-phrase group — creating antecedent/consequent pairs
    that repeat (ABAB) rather than doubling (AABB).

    Skips non-first occurrences of a section label so that phrase_rep
    doesn't scramble bars that _apply_aba_recapitulation later overwrites.
    """
    import random
    result = [list(b) for b in bar_events_list]
    n      = len(result)
    i      = 0
    seen_sections = set()
    while i < n:
        section = section_labels[i]
        j = i
        while j < n and section_labels[j] == section:
            j += 1
        # Only apply phrase repetition to the first run of each section label.
        # Recapitulated sections (second A in ABA, second B in arch, etc.) already
        # have their note content set by _apply_aba_recapitulation; applying phrase_rep
        # on top would overwrite that content with stale pre-recapitulation bars.
        if section in seen_sections:
            i = j
            continue
        seen_sections.add(section)
        # Collect phrase start indices within this section
        phrases = []
        p = i
        while p < j:
            phrases.append(p)
            p = min(p + phrase_bars, j)
        # Walk in 4-phrase groups: copy first pair over second pair (ABAB)
        for g in range(0, len(phrases), 4):
            if g + 2 >= len(phrases):
                break                   # need at least 3 phrases to do anything
            if random.random() >= repeat_prob:
                continue
            for offset in range(2):     # copy phrase g+0 → g+2, phrase g+1 → g+3
                if g + offset + 2 >= len(phrases):
                    break
                src = phrases[g + offset]
                dst = phrases[g + offset + 2]
                src_end = min(src + phrase_bars, j)
                dst_end = min(dst + phrase_bars, j)
                for k in range(min(dst_end - dst, src_end - src)):
                    result[dst + k] = list(result[src + k])
        i = j
    return result


# Chord-tone intervals by quality index (matches QUALITIES list in tokenizer)
_QUALITY_INTERVALS = [
    [0, 4, 7],       # 0 maj
    [0, 3, 7],       # 1 min
    [0, 3, 6],       # 2 dim
    [0, 4, 8],       # 3 aug
    [0, 4, 7, 10],   # 4 dom7
    [0, 4, 7, 11],   # 5 maj7
    [0, 3, 7, 10],   # 6 min7
    [0, 3, 6, 10],   # 7 hdim7
    [0, 2, 7],       # 8 sus2
    [0, 5, 7],       # 9 sus4
]


def _enforce_cadences(chords, section_labels, phrase_bars, tonic_root,
                      dominant_qual=4, tonic_qual_override=None):
    """
    Post-process the chord sequence so the piece resolves cleanly at the end
    without overwriting the model's mid-phrase harmonic choices.
    """
    if phrase_bars < 2 or not chords:
        return chords
    result = list(chords)
    n      = len(result)
    dom    = (tonic_root + 7) % 12
    tonic_qual  = tonic_qual_override
    if tonic_qual is None:
        qual_counts = {}
        for c in chords:
            if c is None:
                continue
            qual_counts[c[1]] = qual_counts.get(c[1], 0) + 1
        tonic_qual = 1 if qual_counts.get(1, 0) + qual_counts.get(6, 0) \
                         > qual_counts.get(0, 0) + qual_counts.get(5, 0) else 0

    # Find the last A-section bar so the piece always ends on I.
    last_a_bar = max((i for i in range(n) if section_labels[i] in ('A', 'none')), default=-1)

    if last_a_bar >= 0:
        result[last_a_bar] = (tonic_root, tonic_qual)
        if last_a_bar - 1 >= 0 and section_labels[last_a_bar - 1] in ('A', 'none'):
            result[last_a_bar - 1] = (dom, dominant_qual)
    return result


def _impose_lh_pattern(bar_events, chord, bass_split=58, lh_vel=3, pattern='alberti',
                       bar_idx=0, next_chord=None):
    """
    Replace model-generated bass notes (pitch < bass_split) with a left-hand
    pattern built from the bar's chord tones.
    Treble content (pitch >= bass_split) is preserved unchanged.

    Patterns (all in 4/4, positions 0-15 = 16th-note grid):
      alberti : 8th-note broken chord — root,5th,3rd,5th × 2 per bar
      stride  : long bass note + staccato chord cluster on off-beats
      murky   : sustained root with punched octave on beats 2 & 4
      waltz   : long root + short chord on beats 2 & 3
      walking : stepwise quarter-note bass line through chord/scale tones

    Each pattern procedurally varies note choices bar-to-bar (bass register,
    chord voicing, passing tones) while keeping its rhythmic skeleton.
    bar_idx seeds per-bar variation so MIDI is reproducible.
    next_chord (root, qual) used for chromatic approach tones.
    """
    if chord is None:
        return bar_events
    root, qual = chord
    ivs = _QUALITY_INTERVALS[qual] if qual < len(_QUALITY_INTERVALS) else [0, 4, 7]
    bass_root = 36 + root

    def _cl(p):
        """Clamp pitch into bass register [21, bass_split)."""
        while p >= bass_split: p -= 12
        while p < 21: p += 12
        return p

    third  = _cl(bass_root + 12 + (ivs[1] if len(ivs) > 1 else 4))
    fifth  = _cl(bass_root + 12 + (ivs[-1] if len(ivs) > 1 else 7))
    seventh = _cl(bass_root + 12 + ivs[3]) if len(ivs) > 3 else None
    bass_oct = _cl(bass_root + 12)          # root one octave up (still below split)
    bass_low = bass_root - 12 if bass_root - 12 >= 21 else bass_root

    v0 = lh_vel
    v1 = max(lh_vel - 1, 0)
    v2 = max(lh_vel - 2, 0)

    _rng = random.Random(bar_idx * 31 + root * 7 + qual)

    if pattern == 'stride':
        # Beat-1 bass: root (usual) or lower octave root — chord tones only
        b1_low = bass_root - 12 if bass_root - 12 >= 21 else bass_root
        b1_pool = [bass_root, b1_low]
        b1_w    = [6, 2]
        b1 = _rng.choices(b1_pool, weights=b1_w)[0]

        # Beat-3 bass: 5th of chord, or root (chord tones only)
        b3_pool = [fifth, bass_root]
        b3 = _rng.choices(b3_pool, weights=[6, 2])[0]
        b3 = max(21, min(b3, bass_split - 1))

        # Off-beat voicings — beats 2 and 4 chosen independently
        sev = seventh if seventh else fifth
        v_opts = [[third, fifth], [third, fifth], [third, sev], [fifth, sev], [third], [fifth]]
        b2_v = _rng.choice(v_opts)
        b4_v = _rng.choice(v_opts)

        # Bass notes are quarter notes (dur=4) so they end exactly at the next beat
        lh = [(0, b1, 4, v0), (8, b3, 4, v0)]
        for p in b2_v:
            lh.append((4,  p, 2, v1))
        for p in b4_v:
            lh.append((12, p, 2, v1))

    elif pattern == 'murky':
        # Upper octave companion: root octave or 5th depending on bar
        companion = bass_oct if _rng.random() < 0.75 else fifth
        companion = max(21, min(companion, bass_split - 1))

        # Second-half root: root (common) or 5th (colour change)
        b3_root = fifth if _rng.random() < 0.3 else bass_root
        b3_root = max(21, min(b3_root, bass_split - 1))

        lh = [
            (0,  bass_root,  8, v0),
            (4,  companion,  2, v2),
            (8,  b3_root,    4, v0),
            (12, companion,  2, v2),
        ]
        # Occasional extra octave hit on "and of 2" (pos 6) for busier bars
        if _rng.random() < 0.2:
            lh.append((6, companion, 2, v2))

    elif pattern == 'waltz':
        # Beat-1 bass: root at standard octave (common) or one octave lower
        b1_low = bass_root - 12 if bass_root - 12 >= 21 else bass_root
        b1 = _rng.choices([bass_root, b1_low], weights=[6, 2])[0]

        # Chord voicings on beats 2 and 3 — independently chosen each bar
        sev = seventh if seventh else fifth
        v_opts = [
            [third, fifth],   # standard (double-weighted)
            [third, fifth],
            [third],          # open third
            [fifth],          # open fifth
            [third, sev],     # add colour with 7th
        ]
        b2_v = _rng.choice(v_opts)
        b3_v = _rng.choice(v_opts)

        # Occasionally drop beat 3 entirely (hemiola feel)
        skip_b3 = _rng.random() < 0.12

        lh = [(0, b1, 6, v0)]
        for p in b2_v:
            lh.append((4, p, 2, v1))
        if not skip_b3:
            for p in b3_v:
                lh.append((8, p, 2, v1))

        # Occasional chromatic passing note at pos 10 (lead into next bar)
        if next_chord and _rng.random() < 0.2:
            nr = _cl(36 + next_chord[0])
            passing = b1 + (1 if nr > b1 else -1)
            lh.append((10, max(21, min(passing, bass_split - 1)), 2, v2))

    elif pattern == 'walking':
        # Approach tone: chromatic or whole-step into next bar's root
        approach = bass_root
        if next_chord is not None:
            nr = _cl(36 + next_chord[0])
            approach = nr + _rng.choice([-2, -1, 1, 2])
            approach = max(21, min(approach, bass_split - 1))

        # Three motion archetypes, weighted equally
        motion = _rng.choices(['up', 'down', 'arch'], weights=[4, 4, 2])[0]
        if motion == 'up':
            step = _rng.choice([1, 2])
            p2 = bass_root + step
            p3 = bass_root + _rng.choice([3, 4, 5])
            walk = [bass_root, p2, p3, approach if next_chord else bass_root + 7]
        elif motion == 'down':
            step = _rng.choice([1, 2])
            p2 = fifth - step
            p3 = fifth - _rng.choice([3, 4])
            walk = [fifth, p2, p3, approach if next_chord else bass_root]
        else:  # arch: up then back
            peak = bass_root + _rng.choice([4, 5, 7])
            mid  = bass_root + _rng.choice([1, 2])
            walk = [bass_root, peak, mid, approach if next_chord else bass_root]

        walk = [max(21, min(p, bass_split - 1)) for p in walk]
        lh = [
            (0,  walk[0], 4, v0),
            (4,  walk[1], 4, v1),
            (8,  walk[2], 4, v0),
            (12, walk[3], 4, v1),
        ]

    else:  # alberti — 8th notes, real classical style
        # Vary bass register on beat 1 (occasionally dip an octave lower)
        b1 = bass_low if _rng.random() < 0.2 and bass_low < bass_root else bass_root

        # Inner voice variation: swap 3rd/5th, or replace 5th with 7th
        sev = seventh if seventh else fifth
        if _rng.random() < 0.25:
            s3, s5 = fifth, third               # swap inner voices
        elif _rng.random() < 0.15 and seventh:
            s3, s5 = third, sev                 # 7th colours second half
        else:
            s3, s5 = third, fifth

        lh = [
            (0,  b1,        2, v0),
            (2,  fifth,     2, v1),
            (4,  third,     2, v1),
            (6,  fifth,     2, v1),
            (8,  bass_root, 2, v0),             # strong re-attack on beat 3
            (10, fifth,     2, v1),
            (12, s3,        2, v1),
            (14, s5,        2, v1),
        ]

    # Keep everything above bass_split; in the inner-voice zone (bass_split-12 to bass_split)
    # only keep chord tones — passing/chromatic notes there clash with the bass pattern.
    inner_floor = max(bass_split - 8, 24)
    chord_pcs   = set((root + iv) % 12 for iv in ivs)
    treble = [
        (pos, p, d, v) for pos, p, d, v in bar_events
        if p >= bass_split or (p >= inner_floor and p % 12 in chord_pcs)
    ]

    # For waltz: at beat positions where the chord lands (pos 4 and 8 = beats 2 and 3),
    # filter treble notes to chord tones only.  Non-chord-tone treble notes at these
    # exact positions clash directly with the chord hit in the bass.
    if pattern == 'waltz':
        chord_hit_positions = {4, 8}
        filtered = []
        for pos, p, d, v in treble:
            if pos in chord_hit_positions and p % 12 not in chord_pcs:
                # Non-chord-tone on a chord-hit beat — drop it
                pass
            else:
                filtered.append((pos, p, d, v))
        # Only use filtered version if it still has some treble notes
        if filtered:
            treble = filtered

    return treble + lh


# ── Procedural bass generation ────────────────────────────────────────────────

def _proc_chord_pitches(chord, bass_split, low=24):
    """All chord-tone pitches available in the bass register."""
    root, qual = chord
    ivs = _QUALITY_INTERVALS[qual] if qual < len(_QUALITY_INTERVALS) else [0, 4, 7]
    out = set()
    for iv in ivs:
        p = 36 + root + iv          # start in C2 octave
        while p >= bass_split: p -= 12
        while p < low: p += 12
        out.add(p)
        if p - 12 >= low:           # also one octave lower when in range
            out.add(p - 12)
    return sorted(out)


def _proc_scale_pitches(key_root, bass_split, low=24):
    """All diatonic scale pitches in the bass register."""
    pcs = {(key_root + iv) % 12 for iv in _MAJOR_INTERVALS}
    return [p for p in range(low, bass_split) if p % 12 in pcs]


def _proc_pick(target, candidates, rng, spread=8.0):
    """
    Pick a pitch from candidates using a Gaussian preference centred on target.
    Candidates close to target are strongly preferred; distant ones rarely chosen.
    """
    if not candidates:
        return target
    weights = [math.exp(-((p - target) ** 2) / max(spread ** 2, 1.0))
               for p in candidates]
    total = sum(weights)
    if total < 1e-12:
        return rng.choice(candidates)
    r = rng.random() * total
    acc = 0.0
    for p, w in zip(candidates, weights):
        acc += w
        if acc >= r:
            return p
    return candidates[-1]


def _generate_bass_bar(chord, bass_split=58, lh_vel=3, bar_idx=0,
                       next_chord=None, prev_bass=None,
                       phrase_pos=0, phrase_bars=4, key_root=None):
    """
    Procedurally compose a bass line for one bar.

    Core technique: weighted random walk over chord/scale/chromatic pitches,
    driven by a per-bar rhythm template chosen according to phrase position.
    Voice leading bias (Gaussian proximity) keeps motion smooth across bars.
    Approach tones connect bars at chord changes.

    Returns a list of (pos, pitch, dur, vel) events; all pitches < bass_split.
    """
    if chord is None:
        return []

    rng = random.Random(bar_idx * 53 + chord[0] * 11 + chord[1] * 3)

    root, _ = chord
    bass_root_pc = (36 + root) % 12

    chord_ps = _proc_chord_pitches(chord, bass_split)
    root_ps  = [p for p in chord_ps if p % 12 == bass_root_pc]

    if key_root is not None:
        scale_ps = _proc_scale_pitches(key_root, bass_split)
    else:
        scale_ps = list(range(24, bass_split))          # chromatic when no key

    # Chromatic passing tones: everything in range
    all_ps = list(range(24, bass_split))

    # Starting pitch: voice-lead from previous bar's last bass note
    cur = prev_bass if prev_bass is not None else (root_ps[0] if root_ps else chord_ps[0])

    # ── Rhythm template selection ────────────────────────────────────────────
    # Each template is a list of (pos, dur, beat_weight):
    #   beat_weight 2 = downbeat  (root preferred, voice-led close)
    #   beat_weight 1 = beat      (chord tone preferred)
    #   beat_weight 0 = weak/pass (scale or chromatic passing tone)
    TEMPLATES = [
        # Sustained / minimal
        [(0, 16, 2)],                                          # whole note
        [(0, 8, 2), (8, 8, 1)],                               # 2 halves
        [(0, 12, 2), (12, 4, 1)],                             # dotted half + q
        # Quarter-note walking
        [(0, 4, 2), (4, 4, 1), (8, 4, 2), (12, 4, 1)],       # 4 quarters
        [(0, 4, 2), (4, 4, 0), (8, 4, 2), (12, 4, 0)],       # alt-beat quarters
        # Mixed / syncopated
        [(0, 6, 2), (6, 2, 0), (8, 4, 1), (12, 4, 1)],       # d.q + 8th + q + q
        [(0, 4, 2), (4, 2, 0), (6, 2, 0), (8, 4, 1), (12, 4, 1)],  # q+2×8th+q+q
        [(0, 8, 2), (8, 4, 1), (12, 4, 1)],                  # half + q + q
        [(0, 4, 2), (4, 4, 1), (8, 8, 2)],                   # q + q + half
        [(0, 4, 2), (4, 8, 0), (12, 4, 1)],                  # q + half(pass) + q
        # Active / walking
        [(0, 4, 2), (4, 2, 0), (6, 2, 0), (8, 2, 1), (10, 2, 0), (12, 4, 1)],
        [(0, 2, 2), (2, 2, 0), (4, 2, 1), (6, 2, 0), (8, 2, 2), (10, 2, 0), (12, 4, 1)],
    ]

    # Activity level: arch within phrase (low at ends, peak at midpoint)
    pf = phrase_pos / max(phrase_bars - 1, 1) if phrase_bars > 1 else 0.5
    activity = 0.15 + 0.85 * math.sin(pf * math.pi)

    t_weights = []
    for t in TEMPLATES:
        n = len(t)
        if n == 1:   w = max(0.05, 1.2 - activity * 1.1)
        elif n == 2: w = max(0.1,  1.0 - activity * 0.7)
        elif n == 3: w = 0.5 + activity * 0.3
        elif n <= 5: w = 0.3 + activity * 0.8
        else:        w = activity * 0.9
        t_weights.append(w)

    total_w = sum(t_weights)
    t_weights = [w / total_w for w in t_weights]
    template = rng.choices(TEMPLATES, weights=t_weights)[0]

    # ── Generate a pitch for each rhythmic event ─────────────────────────────
    notes = []
    for i, (pos, dur, beat_w) in enumerate(template):
        is_first = (i == 0)
        is_last  = (i == len(template) - 1)

        if is_first:
            # Beat 1: root strongly preferred, voice-led from previous bar
            if root_ps and rng.random() < 0.70:
                pitch = min(root_ps, key=lambda p: abs(p - cur))
            else:
                pitch = _proc_pick(cur, chord_ps, rng, spread=6.0)

        elif is_last and next_chord is not None and rng.random() < 0.60:
            # Approach tone to next bar's root: half- or whole-step
            nr = 36 + next_chord[0]
            while nr >= bass_split: nr -= 12
            while nr < 24: nr += 12
            step = rng.choice([-1, 1, -2, 2])
            pitch = max(24, min(nr + step, bass_split - 1))

        elif beat_w >= 2:
            # Downbeat: chord tone, voice-led
            pitch = _proc_pick(cur, chord_ps, rng, spread=7.0)

        elif beat_w == 1:
            # Beat: chord or nearby scale tone
            cands = list(set(chord_ps + [p for p in scale_ps if abs(p - cur) <= 7]))
            pitch = _proc_pick(cur, cands if cands else chord_ps, rng, spread=5.0)

        else:
            # Weak / passing: diatonic scale tones within 3 semitones (no chromatic)
            cands = [p for p in scale_ps if abs(p - cur) <= 3]
            if not cands:
                cands = [p for p in scale_ps if abs(p - cur) <= 5] or chord_ps
            pitch = _proc_pick(cur, cands, rng, spread=2.0)

        vel = lh_vel if beat_w >= 2 else max(lh_vel - 1, 0)
        notes.append((pos, pitch, dur, vel))
        cur = pitch

    return notes


def _apply_procedural_bass(bar_events_list, chords, bass_split=58, lh_vel=3,
                           phrase_bars=4, key_root=None):
    """
    Run procedural bass generation over all bars, passing voice-leading state
    (last bass pitch) from each bar to the next.
    """
    result = []
    prev_bass = None
    for i, events in enumerate(bar_events_list):
        chord    = chords[i] if i < len(chords) else None
        next_ch  = chords[i + 1] if i + 1 < len(chords) else None
        phrase_pos = i % phrase_bars

        inner_floor = max(bass_split - 8, 24)
        if chord is not None:
            _ivs = _QUALITY_INTERVALS[chord[1]] if chord[1] < len(_QUALITY_INTERVALS) else [0, 4, 7]
            chord_pcs = set((chord[0] + iv) % 12 for iv in _ivs)
        else:
            chord_pcs = set(range(12))
        treble = [
            (pos, p, d, v) for pos, p, d, v in events
            if p >= bass_split or (p >= inner_floor and p % 12 in chord_pcs)
        ]
        bass   = _generate_bass_bar(
            chord, bass_split=bass_split, lh_vel=lh_vel,
            bar_idx=i, next_chord=next_ch, prev_bass=prev_bass,
            phrase_pos=phrase_pos, phrase_bars=phrase_bars,
            key_root=key_root,
        )
        if bass:
            prev_bass = bass[-1][1]     # last event's pitch, feeds next bar

        result.append(treble + bass)
    return result


def _thin_treble_texture(bar_events_list, bass_split=58, max_voices=2, inner_max_dur=6):
    """
    Thin the treble chord texture to at most max_voices simultaneous onsets per
    grid position (highest pitches kept — melody first, then closest harmony).
    Also caps inner-voice (non-top) note durations at inner_max_dur 16th notes
    so they don't sustain as a pad under a moving melody line.
    """
    result = []
    for events in bar_events_list:
        # Group treble indices by onset position
        by_pos = {}
        for i, (pos, pitch, dur, vel) in enumerate(events):
            if pitch >= bass_split:
                by_pos.setdefault(pos, []).append(i)

        keep   = set(range(len(events)))   # start: keep everything
        capped = {}                        # index -> new_dur for inner voices

        for pos, indices in by_pos.items():
            if len(indices) <= max_voices:
                # Still cap inner-voice durations even when not thinning
                if len(indices) == max_voices:
                    by_pitch = sorted(indices, key=lambda i: events[i][1], reverse=True)
                    for i in by_pitch[1:]:
                        pos_, p_, dur_, vel_ = events[i]
                        if dur_ > inner_max_dur:
                            capped[i] = inner_max_dur
                continue
            # Sort by pitch descending: keep top max_voices, discard rest
            by_pitch = sorted(indices, key=lambda i: events[i][1], reverse=True)
            for i in by_pitch[max_voices:]:
                keep.discard(i)
            # Cap inner voices
            for i in by_pitch[1:max_voices]:
                pos_, p_, dur_, vel_ = events[i]
                if dur_ > inner_max_dur:
                    capped[i] = inner_max_dur

        bar = []
        for i, ev in enumerate(events):
            if i not in keep:
                continue
            if i in capped:
                pos_, p_, dur_, vel_ = ev
                bar.append((pos_, p_, capped[i], vel_))
            else:
                bar.append(ev)
        result.append(bar)
    return result


def _ensure_treble_presence(bar_events_list, chords, bass_split=58, phrase_bars=4,
                            section_labels=None):
    """
    Keep every bar connected to a treble line.

    Some model bars contain only low-register material; after the left-hand pass
    those bars become accompaniment-only, which sounds like the melody vanished.
    Add a conservative chord-tone anchor in the treble when a bar has no treble.
    """
    result = [list(bar) for bar in bar_events_list]
    prev_top = 72
    prev_section = None

    for i, events in enumerate(result):
        # Reset register anchor at section boundaries so inserted notes don't
        # inherit a register from a contrasting section.
        if section_labels is not None:
            curr_section = section_labels[i] if i < len(section_labels) else None
            if curr_section != prev_section:
                prev_top = 72
            prev_section = curr_section

        treble = [p for _, p, _, _ in events if p >= bass_split]
        if treble:
            prev_top = max(treble)
            continue

        chord = chords[i] if i < len(chords) else None
        if chord is None:
            continue
        root, qual = chord
        ivs = _QUALITY_INTERVALS[qual] if qual < len(_QUALITY_INTERVALS) else [0, 4, 7]
        candidates = []
        for octave_root in (48, 60, 72):
            base = octave_root + root
            for iv in ivs[:3]:
                p = base + iv
                if bass_split <= p <= 88:
                    candidates.append(p)
        if not candidates:
            continue

        pitch = min(candidates, key=lambda p: abs(p - prev_top))
        phrase_pos = i % max(phrase_bars, 1)
        pos = 0 if phrase_pos == 0 else 8
        result[i].append((pos, pitch, 8, min(N_VEL_BINS - 1, 5)))
        prev_top = pitch

    return result


def _truncate_cross_chord_sustains(bar_events_list, chords, bass_split=58):
    """
    Treble notes with long durations can sustain across bar boundaries into a bar
    where they are no longer chord tones, creating lingering dissonance.
    For each bar transition:
    1. Truncate any carry that is a non-chord-tone in the next bar.
    2. Truncate any carry that creates a m2 semitone clash with any note in the
       next bar (even if the carry is a chord tone).
    """
    if not chords:
        return bar_events_list
    result = [list(bar) for bar in bar_events_list]
    for bar_idx in range(1, len(result)):
        chord = chords[bar_idx] if bar_idx < len(chords) else None
        next_bar = result[bar_idx]
        prev = result[bar_idx - 1]
        chord_pcs = set()
        if chord is not None:
            root, qual = chord
            ivs = _QUALITY_INTERVALS[qual] if qual < len(_QUALITY_INTERVALS) else [0, 4, 7]
            chord_pcs = set((root + iv) % 12 for iv in ivs)
        for j, (pos, p, d, v) in enumerate(prev):
            if p < bass_split or pos + d <= 16:
                continue
            # Condition 1: non-chord-tone in next bar
            if chord_pcs and p % 12 not in chord_pcs:
                result[bar_idx - 1][j] = (pos, p, max(1, 16 - pos), v)
                continue
            # Condition 2: m2 clash with any treble note in the next bar
            if any(abs(p - p2) == 1 and p2 >= bass_split
                   for pos2, p2, d2, v2 in next_bar):
                result[bar_idx - 1][j] = (pos, p, max(1, 16 - pos), v)
    return result


def _polish_top_voice(bar_events_list, bass_split=58, min_onset_gap=2, same_pitch_threshold=3):
    """
    Fix the two main improv-sloppy top-voice patterns, while preserving harmonic texture:

    1. Enforce a minimum gap of min_onset_gap (default 2 = 8th note) between consecutive
       top-voice onsets. Positions that are too close get their top note absorbed into the
       previous top note (duration extended). This eliminates trill/tremolo patterns and
       rapid 16th-note alternation in the melody line.

    2. Collapse runs of >= same_pitch_threshold consecutive same-pitch top-voice onsets
       into one sustained note (e.g. 4× the same note → one held note).

    Crucially: inner voices at ALL positions are preserved untouched, so the harmonic
    richness and natural density of the accompaniment is not disturbed.
    """
    result = []
    for events in bar_events_list:
        bass   = [(pos, p, d, v) for pos, p, d, v in events if p < bass_split]
        treble = [(pos, p, d, v) for pos, p, d, v in events if p >= bass_split]

        if not treble:
            result.append(events)
            continue

        # Original top note at each onset position
        orig_top = {}
        for pos, p, d, v in treble:
            if pos not in orig_top or p > orig_top[pos][0]:
                orig_top[pos] = (p, d, v)

        positions = sorted(orig_top.keys())

        # Walk positions and build a cleaned top-voice list.
        # Two rules, applied together at each step:
        #   a) too close (gap < min_onset_gap)  → absorb into previous
        #   b) same pitch as previous (consecutive same-pitch run ≥ threshold) → absorb
        # Rule (b) is tracked as a run counter; we only absorb when the run is long enough.
        top_notes = []  # [pos, pitch, dur, vel] — mutable entries
        same_run  = 1   # length of current same-pitch run (counting from top_notes[-1])

        for pos in positions:
            pitch, dur, vel = orig_top[pos]

            if not top_notes:
                top_notes.append([pos, pitch, dur, vel])
                same_run = 1
                continue

            last = top_notes[-1]
            too_close  = (pos - last[0]) < min_onset_gap
            same_pitch = (pitch == last[1])

            if same_pitch:
                same_run += 1
            else:
                same_run = 1

            absorb = too_close or (same_pitch and same_run >= same_pitch_threshold)

            if absorb:
                # Extend the previous top note to cover this position's original duration
                last[2] = max(last[2], pos - last[0] + dur)
            else:
                top_notes.append([pos, pitch, dur, vel])

        # Build final event list:
        # - All INNER-VOICE treble notes (not the top at their position) are kept as-is
        # - Top-voice notes use the (possibly extended) durations from top_notes
        top_dur_at = {tn[0]: tn[2] for tn in top_notes}          # pos → new dur
        top_kept   = {tn[0] for tn in top_notes}                  # positions with a kept top

        final = []
        for pos, p, d, v in treble:
            is_top = (p == orig_top[pos][0])
            if is_top:
                if pos in top_kept:
                    final.append((pos, p, top_dur_at[pos], v))
                # else: this top note was absorbed — drop it from the top voice
            else:
                # Inner voice: always keep
                final.append((pos, p, d, v))

        result.append(bass + final)
    return result


def _collapse_bass_runs(bar_events_list, bass_split=58, same_pitch_threshold=3):
    """
    Collapse runs of >=same_pitch_threshold consecutive same-pitch bass onsets
    into one sustained note.  Treble notes are untouched.
    """
    result = []
    for events in bar_events_list:
        bass   = sorted([(pos, p, d, v) for pos, p, d, v in events if p < bass_split],
                        key=lambda e: e[0])
        treble = [(pos, p, d, v) for pos, p, d, v in events if p >= bass_split]

        if not bass:
            result.append(events)
            continue

        cleaned = []
        run_pitch = None
        run_count = 0
        for pos, p, d, v in bass:
            if p == run_pitch:
                run_count += 1
                if run_count < same_pitch_threshold:
                    cleaned.append((pos, p, d, v))
                else:
                    # extend duration of the first note in the run instead
                    prev = cleaned[-1] if cleaned else None
                    if prev and prev[1] == p:
                        cleaned[-1] = (prev[0], prev[1], pos - prev[0] + d, prev[3])
            else:
                run_pitch = p
                run_count = 1
                cleaned.append((pos, p, d, v))

        result.append(cleaned + treble)
    return result


def _collapse_inner_voice_runs(bar_events_list, bass_split=58, same_pitch_threshold=3):
    """
    Collapse repeated same-pitch runs in inner treble voices (>= bass_split, not
    the highest note at their onset position).  Tracks each pitch independently:
    if the same pitch appears at >= same_pitch_threshold consecutive positions,
    collapse into a single sustained note at the first position.
    Top voice and bass are untouched.
    """
    from collections import defaultdict

    result = []
    for events in bar_events_list:
        treble = [(pos, p, d, v) for pos, p, d, v in events if p >= bass_split]
        bass   = [(pos, p, d, v) for pos, p, d, v in events if p < bass_split]

        if not treble:
            result.append(events)
            continue

        # Highest pitch at each position = top voice (untouched)
        top_at = {}
        for pos, p, d, v in treble:
            if pos not in top_at or p > top_at[pos]:
                top_at[pos] = p

        top_events   = [(pos, p, d, v) for pos, p, d, v in treble if p == top_at[pos]]
        inner_events = [(pos, p, d, v) for pos, p, d, v in treble if p != top_at[pos]]

        if not inner_events:
            result.append(events)
            continue

        # Group inner events by pitch, sorted by position
        by_pitch = defaultdict(list)
        for pos, p, d, v in sorted(inner_events, key=lambda e: e[0]):
            by_pitch[p].append((pos, d, v))

        kept_inner = []
        for pitch, pev in by_pitch.items():
            # Find and collapse runs of consecutive positions (pos_i+1 == pos_i+1)
            out = []
            i = 0
            while i < len(pev):
                # Extend run as far as positions are consecutive
                j = i
                while j + 1 < len(pev) and pev[j + 1][0] == pev[j][0] + 1:
                    j += 1
                run_len = j - i + 1
                if run_len >= same_pitch_threshold:
                    # Collapse: one note at run start with duration spanning the whole run
                    s_pos, s_dur, s_vel = pev[i]
                    e_pos, e_dur, _     = pev[j]
                    new_dur = e_pos - s_pos + e_dur
                    out.append((s_pos, pitch, new_dur, s_vel))
                else:
                    for k in range(i, j + 1):
                        pos, dur, vel = pev[k]
                        out.append((pos, pitch, dur, vel))
                i = j + 1
            kept_inner.extend(out)

        result.append(bass + top_events + kept_inner)
    return result


def _break_top_voice_loops(bar_events_list, bass_split=58, run_threshold=3):
    """
    Detect runs of >= run_threshold consecutive bars sharing the same top-voice
    pitch and octave-displace the repeated bars to break the loop.

    Only bars 2+ within a run are displaced (the first bar keeps its pitch).
    Displacement tries -12 first (lower octave), then +12, staying within
    [bass_split, 91].  If no valid displacement exists, the bar is left alone.
    """
    MELODY_LO = bass_split
    MELODY_HI = 91

    n = len(bar_events_list)
    result = [list(bar) for bar in bar_events_list]

    # Find top note per bar
    def bar_top(bar):
        pitches = [p for _, p, _, _ in bar if p >= bass_split]
        return max(pitches) if pitches else None

    for _pass in range(4):   # iterate until no new loops are created
        tops = [bar_top(result[b]) for b in range(n)]
        changed = False

        b = 0
        while b < n:
            if tops[b] is None:
                b += 1
                continue
            run_end = b + 1
            while run_end < n and tops[run_end] == tops[b]:
                run_end += 1
            run_len = run_end - b
            if run_len >= run_threshold:
                orig_pitch = tops[b]
                for rb in range(b + 1, run_end):
                    best_delta = None
                    for delta in (-12, 12, -24, 24):
                        cand = orig_pitch + delta
                        if MELODY_LO <= cand <= MELODY_HI:
                            best_delta = delta
                            break
                    if best_delta is None:
                        continue
                    new_bar = []
                    for pos, p, d, v in result[rb]:
                        if p == orig_pitch:
                            new_bar.append((pos, p + best_delta, d, v))
                        else:
                            new_bar.append((pos, p, d, v))
                    result[rb] = new_bar
                    tops[rb] = orig_pitch + best_delta
                    changed = True
            b = run_end

        if not changed:
            break

    return result


def _smooth_melody_leaps(bar_events_list, bass_split=58, max_leap=7, section_labels=None):
    """
    Reduce large leaps in the top (melody) voice by octave-displacing notes.

    When the highest treble note at grid position P leaps more than max_leap
    semitones from the previous melody note, try ±12 (and ±24 as fallback)
    to find a register that cuts the interval.  The pitch class is unchanged,
    so harmony is preserved.  Inner voices are untouched.

    prev_top is NOT reset at section boundaries so the smoother bridges the
    gap between sections — the first note of B is pulled toward where A ended.
    """
    MELODY_LO = bass_split    # never push melody below the treble boundary
    MELODY_HI = 91            # ~G6, a comfortable piano melody ceiling

    result = [list(bar) for bar in bar_events_list]
    prev_top = None

    for bar_idx, events in enumerate(result):
        # Pre-pass: clamp EVERY treble note above MELODY_HI (not just the top).
        # A single-top ceiling check misses inner-voice notes that exceed the
        # ceiling and can later form harsh intervals with displaced top notes.
        for i, (pos, pitch, dur, vel) in enumerate(result[bar_idx]):
            if pitch > MELODY_HI:
                clamped = pitch
                for delta in (-12, -24):
                    cand = pitch + delta
                    if MELODY_LO <= cand <= MELODY_HI:
                        clamped = cand
                        break
                if clamped != pitch:
                    result[bar_idx][i] = (pos, clamped, dur, vel)

        # At section boundaries, relax the leap threshold rather than hard-resetting
        # prev_top. A full reset (prev_top=None) allows unconstrained leaps that
        # sound abrupt; keeping prev_top with a wider allowance lets the melody
        # make a deliberate register shift while still preventing extreme jumps.
        at_boundary = (section_labels is not None and bar_idx > 0 and
                       section_labels[bar_idx] != section_labels[bar_idx - 1])
        effective_max_leap = max_leap + 4 if at_boundary else max_leap

        # Build pos → [(event_index, pitch)] for treble notes
        by_pos = {}
        for i, (pos, pitch, dur, vel) in enumerate(result[bar_idx]):
            if pitch >= bass_split:
                by_pos.setdefault(pos, []).append((i, pitch))

        for pos in sorted(by_pos):
            top_idx, top_pitch = max(by_pos[pos], key=lambda x: x[1])

            if prev_top is not None:
                leap = abs(top_pitch - prev_top)
                leap_pc = leap % 12
                # Smooth both large leaps AND tritone intervals at any distance
                if leap > effective_max_leap or leap_pc == 6:
                    best_pitch, best_leap = top_pitch, leap
                    for delta in (-12, 12, -24, 24):
                        cand = top_pitch + delta
                        if MELODY_LO <= cand <= MELODY_HI:
                            cand_leap = abs(cand - prev_top)
                            if cand_leap < best_leap and cand_leap % 12 != 6:
                                best_leap, best_pitch = cand_leap, cand
                    if best_pitch != top_pitch:
                        p_, _, d_, v_ = result[bar_idx][top_idx]
                        result[bar_idx][top_idx] = (p_, best_pitch, d_, v_)
                        top_pitch = best_pitch

            prev_top = top_pitch

    return result


def _smooth_intra_bar_leaps(bar_events_list, diatonic_pcs, bass_split=58):
    """
    Within each bar, smooth harsh melodic intervals (tritone=6, M7=11, m2=1)
    between consecutive top-voice positions by octave-displacing the incoming note.
    Non-diatonic notes that cannot be resolved by displacement are removed entirely.
    Diatonic-to-diatonic harsh intervals are left alone — they are intentional color.
    """
    MELODY_LO = bass_split
    MELODY_HI = 91

    def _harsh(a, b):
        return (abs(a - b) % 12) in {1, 6, 11}

    result = []
    for events in bar_events_list:
        if not events:
            result.append(events)
            continue

        # Index treble notes by position; track the top note at each position
        by_pos = {}
        for idx, (pos, p, d, v) in enumerate(events):
            if p >= bass_split:
                by_pos.setdefault(pos, []).append(idx)

        remove = set()
        modified = {}   # idx → new pitch

        prev_top = None
        for pos in sorted(by_pos):
            indices = by_pos[pos]
            top_idx = max(indices, key=lambda k: events[k][1])
            top_pitch = events[top_idx][1]

            if prev_top is not None and _harsh(top_pitch, prev_top):
                # Try octave displacement to relieve the harsh interval
                best_pitch = top_pitch
                best_dist = abs(top_pitch - prev_top)
                for delta in (-12, 12, -24, 24):
                    cand = top_pitch + delta
                    if MELODY_LO <= cand <= MELODY_HI and not _harsh(cand, prev_top):
                        dist = abs(cand - prev_top)
                        if dist < best_dist:
                            best_dist = dist
                            best_pitch = cand

                if best_pitch != top_pitch:
                    modified[top_idx] = best_pitch
                    top_pitch = best_pitch
                elif diatonic_pcs and (top_pitch % 12) not in diatonic_pcs:
                    # Can't fix via displacement and the note is non-diatonic → drop it
                    remove.add(top_idx)
                    continue   # don't update prev_top — next note checked against same ref

            prev_top = top_pitch

        new_bar = []
        for idx, ev in enumerate(events):
            if idx in remove:
                continue
            if idx in modified:
                pos_, _, d_, v_ = ev
                new_bar.append((pos_, modified[idx], d_, v_))
            else:
                new_bar.append(ev)
        result.append(new_bar)
    return result


def _apply_aba_recapitulation(bar_events_list, section_labels, vel_scale=0.92,
                              skip_start=0):
    """
    For forms with repeated sections (ABA, arch ABCBA, etc.), copy the first
    run of each repeated section label into subsequent runs so the listener
    hears a genuine melodic recapitulation — the same theme returning —
    rather than freshly generated material that merely shares the same chords.

    skip_start bars at the beginning of each target run are left as freshly
    generated — this avoids copying any feedback-loop repetition that may have
    crept into the opening bars of the first run.  The last two bars are also
    left untouched so the final V7→I cadence is preserved.

    A slight velocity reduction (vel_scale < 1) makes the return feel
    reflective rather than a flat repeat.
    """
    n = len(bar_events_list)

    # Collect section runs: label → list of (start, end) pairs
    runs = {}
    i = 0
    while i < n:
        lbl = section_labels[i]
        j = i
        while j < n and section_labels[j] == lbl:
            j += 1
        runs.setdefault(lbl, []).append((i, j))
        i = j

    result = [list(bar) for bar in bar_events_list]

    for lbl, run_list in runs.items():
        if len(run_list) < 2:
            continue
        first_start, first_end = run_list[0]
        first_len = first_end - first_start

        for run_start, run_end in run_list[1:]:
            target_len = run_end - run_start
            # Leave skip_start bars at the start and 2 bars at the end untouched.
            copy_len = min(first_len - skip_start, target_len - skip_start - 2)
            if copy_len <= 0:
                continue
            for offset in range(copy_len):
                src_bar = result[first_start + skip_start + offset]
                result[run_start + skip_start + offset] = [
                    (pos, p, d, max(0, min(N_VEL_BINS - 1, int(v * vel_scale))))
                    for pos, p, d, v in src_bar
                ]

    return result


def _add_cross_bar_ties(bar_events_list, chords, bass_split=58, tie_dur=2,
                        section_labels=None):
    """
    Extend the last melody note of each bar across the barline when it is a
    chord tone in bar N+1.  Without this every note ends before the barline,
    creating dead silence and a hard restart at every bar boundary.

    tie_dur: extra 16th notes sustained into the next bar (default 2 = half beat).
    Only ties if the pitch is a strict chord tone in bar N+1, so the sustained
    note is always harmonically consonant.  Never ties across section boundaries
    (A→B, B→C, etc.) so section contrasts are not blurred.
    """
    result = [list(bar) for bar in bar_events_list]
    n = len(result)

    for bar_idx in range(n - 1):
        # Never tie across a section boundary — would blur the section contrast.
        if section_labels is not None and \
                section_labels[bar_idx] != section_labels[bar_idx + 1]:
            continue
        next_chord = chords[bar_idx + 1] if bar_idx + 1 < len(chords) else None
        if next_chord is None:
            continue
        next_root, next_qual = next_chord
        ivs = _QUALITY_INTERVALS[next_qual] if next_qual < len(_QUALITY_INTERVALS) else [0, 4, 7]
        next_pcs = set((next_root + iv) % 12 for iv in ivs)

        # Collect treble notes in this bar
        treble = [(i, pos, p, d, v)
                  for i, (pos, p, d, v) in enumerate(result[bar_idx])
                  if p >= bass_split]
        if not treble:
            continue

        # Find the top voice note at the latest position
        last_pos = max(pos for _, pos, _, _, _ in treble)
        at_last  = [(i, pos, p, d, v) for i, pos, p, d, v in treble if pos == last_pos]
        i, pos, pitch, dur, vel = max(at_last, key=lambda x: x[2])  # highest pitch

        # Skip if: already crosses, not a chord tone in next bar, or the exact
        # same pitch fires again on beat 1 of bar N+1 (would cause a MIDI retrigger).
        if pos + dur > 16:   # strictly > 16 already crosses; == 16 ends at barline, can extend
            continue
        # Only tie if the pitch is a strict chord tone in bar N+1.
        # The "within 2 semitones" suspension check covers all 12 pitch classes for
        # any common triad, making it meaningless — every note would be tied.
        pc = pitch % 12
        if pc not in next_pcs:
            continue
        same_pitch_at_beat1 = any(p == pitch and pos2 == 0
                                  for pos2, p, d2, v2 in result[bar_idx + 1])
        if same_pitch_at_beat1:
            continue
        # Also skip if the tie would create a m2 clash with a note at beat 1 of bar N+1
        m2_at_beat1 = any(abs(p - pitch) == 1 and pos2 <= 2
                          for pos2, p, d2, v2 in result[bar_idx + 1] if p >= bass_split)
        if m2_at_beat1:
            continue

        # Extend to sustain tie_dur 16th notes into bar N+1
        new_dur = (16 - pos) + tie_dur
        result[bar_idx][i] = (pos, pitch, new_dur, vel)

    return result


def _apply_post_climax_breath(bar_events_list, section_labels, bass_split=58,
                              top_pct=0.15, decay=(0.72, 0.88)):
    """
    Classical music breathes after a climax — the bar(s) following a peak are
    softer before the music resumes. Without this, a 'wow' moment immediately
    crashes into the next phrase at full volume, losing the thread.

    For each section, find bars in the top top_pct of average melody velocity.
    The next len(decay) bars each have all velocities scaled by the corresponding
    decay factor. Multiple overlapping peaks take the minimum (most reduced) factor.
    """
    if not bar_events_list:
        return bar_events_list

    n = len(bar_events_list)

    # Compute avg TREBLE-only velocity per bar — we only want melody peaks to
    # trigger the breath, not loud bass bars (which have no melody to "lose").
    avg_vel = []
    for events in bar_events_list:
        treble = [v for _, p, _, v in events if p >= bass_split and v > 0]
        avg_vel.append(sum(treble) / len(treble) if treble else 0.0)

    # Per-section threshold so a quiet section's moderate peak isn't penalised
    sections = {}
    for i, lbl in enumerate(section_labels):
        sections.setdefault(lbl, []).append(i)

    decay_map = {}   # bar_idx → scale factor
    for lbl, indices in sections.items():
        # Only include bars that actually have melody content
        vels = [avg_vel[i] for i in indices if avg_vel[i] > 0]
        if not vels:
            continue
        cutoff = sorted(vels)[int(len(vels) * (1 - top_pct))]
        for i in indices:
            if avg_vel[i] >= cutoff:
                for d, factor in enumerate(decay, 1):
                    j = i + d
                    if j < n and section_labels[j] == lbl:
                        # Most-reduced factor wins if peaks overlap.
                        # Never bleed decay across a section boundary —
                        # a new section should enter at full presence.
                        decay_map[j] = min(decay_map.get(j, 1.0), factor)

    result = []
    for i, events in enumerate(bar_events_list):
        if i not in decay_map:
            result.append(events)
            continue
        f = decay_map[i]
        result.append([(pos, p, d, max(0, min(N_VEL_BINS - 1, int(v * f))))
                        for pos, p, d, v in events])
    return result


def _apply_velocity_differentiation(bar_events_list, melody_boost=1, accomp_reduce=1):
    """
    At each rhythmic position, identify the highest-pitch note as the melody voice.
    Boost its velocity by melody_boost bins; reduce all lower notes by accomp_reduce.
    This makes the soprano line audibly louder than the inner voices and bass.
    """
    from collections import defaultdict
    result = []
    for events in bar_events_list:
        if not events:
            result.append(events)
            continue
        by_pos = defaultdict(list)
        for e in events:
            by_pos[e[0]].append(e)
        new_events = []
        for pos in sorted(by_pos):
            top = max(e[1] for e in by_pos[pos])
            for p2, pitch, dur, vel in by_pos[pos]:
                if pitch == top:
                    new_vel = min(vel + melody_boost, N_VEL_BINS - 1)
                else:
                    new_vel = max(vel - accomp_reduce, 0)
                new_events.append((p2, pitch, dur, new_vel))
        result.append(new_events)
    return result


def _apply_section_crossfade(bar_events_list, section_labels, fade_bars=3,
                             out_floor=0.60, in_floor=0.65):
    """
    At each section boundary, apply a brief dynamic breath:
      - Last fade_bars of the outgoing section fade from 1.0 → out_floor
      - First fade_bars of the incoming section ramp from in_floor → 1.0
    This signals the transition intentionally rather than masking it,
    matching what live performers do at section changes.
    """
    if not section_labels:
        return bar_events_list

    n = len(bar_events_list)
    scale = [1.0] * n

    # Walk through and mark bars around each section boundary
    for i in range(1, n):
        if section_labels[i] != section_labels[i - 1]:
            # Fade out: bars leading up to the boundary
            for k in range(fade_bars):
                bar = i - 1 - k
                if bar >= 0 and section_labels[bar] == section_labels[i - 1]:
                    t = k / fade_bars          # 0 at boundary, 1 at fade_bars out
                    scale[bar] = min(scale[bar], 1.0 - (1.0 - out_floor) * (1.0 - t))
            # Fade in: bars just after the boundary
            for k in range(fade_bars):
                bar = i + k
                if bar < n and section_labels[bar] == section_labels[i]:
                    t = k / fade_bars          # 0 at boundary, 1 at fade_bars in
                    scale[bar] = min(scale[bar], in_floor + (1.0 - in_floor) * t)

    result = []
    for i, events in enumerate(bar_events_list):
        s = scale[i]
        if s >= 1.0:
            result.append(events)
        else:
            result.append([
                (pos, p, d, max(0, min(N_VEL_BINS - 1, int(v * s + 0.5))))
                for pos, p, d, v in events
            ])
    return result


def build_motif_bias(motif_pitches, device, strength=0.5):
    """
    Pitch-class coherence bias built from a handful of motif pitches.
    Boosts NOTE_ON tokens whose pitch class (mod 12) appears in the motif,
    octave-invariant so the theme is recognised in any register.
    """
    if not motif_pitches:
        return None
    motif_pcs = {p % 12 for p in motif_pitches}
    bias = torch.zeros(NOTE_VOCAB, device=device)
    for tok in range(NOTE_ON_OFF, _NOTE_ON_END):
        pc = (tok - NOTE_ON_OFF + _MIN_PITCH) % 12
        if pc in motif_pcs:
            bias[tok] = strength
    return bias


def vel_arc_bias(bar_idx, n_bars, strength, device):
    """
    Sinusoidal velocity arc — quiet at start/end, louder at midpoint.
    Biases VEL tokens: positive weight for higher velocities at arc peak.
    """
    import math
    t   = bar_idx / max(n_bars - 1, 1)
    arc = math.sin(t * math.pi)           # 0 → 1 → 0
    bias = torch.zeros(NOTE_VOCAB, device=device)
    for i in range(N_VEL_BINS):
        vel_norm = (i / max(N_VEL_BINS - 1, 1)) - 0.5   # −0.5 to +0.5
        bias[NOTE_VEL_OFF + i] = strength * arc * vel_norm * 2
    return bias


# ── Vertical dissonance filter ────────────────────────────────────────
def _reduce_vertical_dissonance(bar_events_list, diatonic_pcs, chords=None, bass_split=58):
    """
    Detects harsh intervals (m2=1, tritone=6, M7=11) between ANY two treble
    notes that overlap in time (not just same-start-position), and removes
    the less important note from each clashing pair. Uses pitch-class distance
    (mod 12) so compound intervals (18st tritone, 23st M7, 13st m2) are caught.

    Priority for removal: non-diatonic > diatonic non-chord-tone > shorter note.
    Bass notes are never removed. Both-chord-tone pairs are left (intentional
    harmony — e.g. the B–F tritone of a G7 chord must not be stripped).
    Runs unconditionally; falls back to chord-tone-only protection when the
    diatonic key is unknown.
    """

    HARSH = {1, 6, 11}  # pitch-class intervals: m2, tritone, M7

    def harsh_actual(p1, p2):
        return abs(p1 - p2) % 12 in HARSH

    def overlaps(n1, n2):
        # n = (pos, pitch, dur, vel); pos/dur in 16th-note slots
        return n1[0] < n2[0] + n2[2] and n2[0] < n1[0] + n1[2]

    result = []
    for bar_idx, bar in enumerate(bar_events_list):
        chord = chords[bar_idx] if (chords and bar_idx < len(chords)) else None
        if chord is not None:
            chord_ivs = _QUALITY_INTERVALS[chord[1]] if chord[1] < len(_QUALITY_INTERVALS) else [0, 4, 7]
            chord_pcs = {(chord[0] + iv) % 12 for iv in chord_ivs}
        else:
            # No chord data: treat nothing as a protected chord tone so the
            # "both chord tones → leave" exception never fires for unknown chords.
            chord_pcs = set()

        treble = [n for n in bar if n[1] >= bass_split]

        # Also include notes from the previous bar that sustain into this one
        # (cross-bar ties added by _add_cross_bar_ties have pos+dur > 16).
        # Represent them as if they start at pos=0 with their remaining duration.
        # Mark them read-only so we only remove from the current bar.
        BAR_SLOTS = 16
        carry_treble = []
        if bar_idx > 0:
            for n in bar_events_list[bar_idx - 1]:
                if n[1] >= bass_split and n[0] + n[2] > BAR_SLOTS:
                    remain = n[0] + n[2] - BAR_SLOTS
                    carry_treble.append((0, n[1], remain, n[3]))

        remove = set()

        def _check_pair(ni, nj, can_remove_ni=True, can_remove_nj=True):
            """Return id of the note to remove, or None."""
            if not overlaps(ni, nj): return None
            if not harsh_actual(ni[1], nj[1]): return None
            pi, pj = ni[1] % 12, nj[1] % 12
            ni_dia = pi in diatonic_pcs; nj_dia = pj in diatonic_pcs
            ni_ct  = pi in chord_pcs;   nj_ct  = pj in chord_pcs
            if not ni_dia and nj_dia:
                return id(ni) if can_remove_ni else None
            elif not nj_dia and ni_dia:
                return id(nj) if can_remove_nj else None
            elif not ni_ct and nj_ct:
                return id(ni) if can_remove_ni else None
            elif not nj_ct and ni_ct:
                return id(nj) if can_remove_nj else None
            elif ni_ct and nj_ct:
                return None  # both chord tones → intentional harmony, leave it
            else:
                # Neither diatonic nor chord-tone context — remove shorter note
                if ni[2] < nj[2]:
                    return id(ni) if can_remove_ni else id(nj) if can_remove_nj else None
                elif nj[2] < ni[2]:
                    return id(nj) if can_remove_nj else id(ni) if can_remove_ni else None
                elif ni[1] < nj[1]:
                    return id(ni) if can_remove_ni else id(nj) if can_remove_nj else None
                else:
                    return id(nj) if can_remove_nj else id(ni) if can_remove_ni else None

        # Check clashes with carry-over sustained notes from previous bar.
        # Carry notes can't be removed (they're in bar N-1's output already).
        # Only remove the current bar's note if it is non-diatonic — a fresh
        # note that is diatonic (or a chord tone) should never yield to a
        # finishing carry note.
        for ni in treble:
            if id(ni) in remove: continue
            for nc in carry_treble:
                if not overlaps(ni, nc): continue
                if not harsh_actual(ni[1], nc[1]): continue
                pi = ni[1] % 12
                if pi not in diatonic_pcs:
                    remove.add(id(ni)); break
                # diatonic current-bar note vs carry — leave it alone

        for i in range(len(treble)):
            ni = treble[i]
            if id(ni) in remove:
                continue
            for j in range(i + 1, len(treble)):
                nj = treble[j]
                if id(nj) in remove:
                    continue
                victim = _check_pair(ni, nj)
                if victim is not None:
                    remove.add(victim)
                    if victim == id(ni):
                        break

        result.append([n for n in bar if id(n) not in remove])
    return result


# ── Chromatic clash filter ────────────────────────────────────────────
def _soften_chromatic_clashes(bar_events_list, diatonic_pcs, bass_split=58):
    """
    Removes or softens very short non-diatonic treble notes that are almost
    certainly accidental model errors rather than intentional chromatic color:
      - dur == 1 AND non-diatonic → remove (single 16th-note chromatic flash)
      - dur == 2 AND non-diatonic → reduce velocity by 3 bins (soften)
    Bass notes are left untouched — chromatic movement in the bass is normal.
    """
    if not diatonic_pcs:
        return bar_events_list
    result = []
    for bar in bar_events_list:
        new_bar = []
        for pos, p, d, v in bar:
            if p >= bass_split and (p % 12) not in diatonic_pcs:
                if d <= 1:
                    continue                        # drop it entirely
                elif d <= 2:
                    v = max(0, v - 3)              # audibly soften
            new_bar.append((pos, p, d, v))
        result.append(new_bar)
    return result


def _strip_harsh_clashes(bar_events_list, bass_split=58):
    """
    Unconditional safety pass: remove same-bar tritone (6st) and M7 (11st)
    clashes between treble notes that overlap in slot space (pitch-class check,
    so compound intervals like 18st and 23st are also caught). Removes the
    shorter note of each clashing pair; ties broken by removing the lower pitch.
    Runs even when key/chord context is unavailable.
    """
    HARSH_PC = {6, 11}  # tritone, M7 by pitch class (m2 handled by _strip_m2_clashes)
    result = []
    for bar in bar_events_list:
        treble = [n for n in bar if n[1] >= bass_split]
        remove = set()
        for i in range(len(treble)):
            ni = treble[i]
            if id(ni) in remove:
                continue
            for j in range(i + 1, len(treble)):
                nj = treble[j]
                if id(nj) in remove:
                    continue
                if abs(ni[1] - nj[1]) % 12 not in HARSH_PC:
                    continue
                if ni[0] >= nj[0] + nj[2] or nj[0] >= ni[0] + ni[2]:
                    continue  # no slot overlap
                # Remove the shorter note; ties broken by lower pitch
                if ni[2] < nj[2]:
                    remove.add(id(ni)); break
                elif nj[2] < ni[2]:
                    remove.add(id(nj))
                elif ni[1] < nj[1]:
                    remove.add(id(ni)); break
                else:
                    remove.add(id(nj))
        result.append([n for n in bar if id(n) not in remove])
    return result


def _strip_m2_clashes(bar_events_list, bass_split=58):
    """
    Final safety pass: remove any remaining same-bar m2 semitone clashes between
    treble notes that overlap in slot space. Removes the lower-pitched note of each
    clashing pair (higher pitch = melody tone wins). No key knowledge needed.
    """
    result = []
    for bar in bar_events_list:
        treble = [n for n in bar if n[1] >= bass_split]
        remove = set()
        for i in range(len(treble)):
            ni = treble[i]
            if id(ni) in remove:
                continue
            for j in range(i + 1, len(treble)):
                nj = treble[j]
                if id(nj) in remove:
                    continue
                if abs(ni[1] - nj[1]) != 1:
                    continue
                if ni[0] >= nj[0] + nj[2] or nj[0] >= ni[0] + ni[2]:
                    continue  # no slot overlap
                # Remove the lower pitch; if one is much longer, keep it
                if ni[2] > nj[2] * 2:
                    remove.add(id(nj))
                elif nj[2] > ni[2] * 2:
                    remove.add(id(ni)); break
                elif ni[1] < nj[1]:
                    remove.add(id(ni)); break
                else:
                    remove.add(id(nj))
        result.append([n for n in bar if id(n) not in remove])
    return result


# ── Final resolution ──────────────────────────────────────────────────
def _apply_final_resolution(bar_events_list, chords, key_root, tonic_qual,
                             section_labels, bass_split=58, fade_bars=6):
    """
    Shapes the final bars so the piece closes convincingly:
    1. Velocity decrescendo over fade_bars (smooth fade-out)
    2. Penultimate bar: thin to long dominant chord tones (clear V setup)
    3. Last bar: rebuild as a decisive tonic chord — root in bass,
       root/5th/root stacked in treble, all on beat 1, held the full bar
    """
    if not bar_events_list or key_root is None:
        return bar_events_list

    result = [list(bar) for bar in bar_events_list]
    n      = len(result)

    _QUAL_IVS = {
        0: [0, 4, 7],        # maj
        1: [0, 3, 7],        # min
        2: [0, 3, 6],        # dim
        3: [0, 4, 8],        # aug
        4: [0, 4, 7, 10],    # dom7
        5: [0, 4, 7, 11],    # maj7
        6: [0, 3, 7, 10],    # min7
        7: [0, 3, 6, 10],    # hdim7
        8: [0, 2, 7],        # sus2
        9: [0, 5, 7],        # sus4
    }

    def _pcs(root, qual):
        ivs = _QUAL_IVS.get(qual, [0, 4, 7])
        return {(root + i) % 12 for i in ivs}

    tonic_ivs = _QUAL_IVS.get(tonic_qual if tonic_qual is not None else 0, [0, 4, 7])
    tonic_pcs = {(key_root + i) % 12 for i in tonic_ivs}

    def _clamp_bass(p):
        while p >= bass_split: p -= 12
        while p < 21:          p += 12
        return p

    def _clamp_treble(p, lo=None):
        lo = lo or bass_split
        while p < lo:  p += 12
        while p > 108: p -= 12
        return p

    # ── 1. Velocity fade ──────────────────────────────────────────────
    for k in range(min(fade_bars, n)):
        i     = n - 1 - k
        # 1.0 at fade_bars out → 0.45 at last bar
        scale = 0.45 + 0.55 * (k / max(fade_bars - 1, 1))
        result[i] = [
            (pos, p, d, max(0, int(v * scale + 0.5)))
            for pos, p, d, v in result[i]
        ]

    # ── 2. Penultimate bar: thin to long dominant chord tones ─────────
    if n >= 2:
        penu_chord = chords[n - 2] if n - 2 < len(chords) and chords[n - 2] else None
        penu_pcs   = _pcs(*penu_chord) if penu_chord else tonic_pcs
        penu       = result[n - 2]
        # Keep only chord tones with duration >= 2; cap at 5 notes
        keepers = sorted(
            [(pos, p, d, v) for pos, p, d, v in penu
             if (p % 12) in penu_pcs and d >= 2],
            key=lambda x: x[0]
        )
        if len(keepers) >= 2:
            # Extend each keeper to fill from its position to bar end
            keepers = [(pos, p, max(d, min(8, POSITIONS - pos)), v)
                       for pos, p, d, v in keepers[:5]]
            result[n - 2] = keepers

    # ── 3. Last bar: rebuild as a decisive tonic final chord ──────────
    # Infer register from the surrounding bars so the chord sits in
    # the same part of the keyboard the piece has been using.
    ref_bars   = result[max(0, n - 5): n - 1]
    ref_treble = [p for bar in ref_bars for _, p, _, _ in bar if p >= bass_split]
    if ref_treble:
        ref_mid = int(sum(ref_treble) / len(ref_treble))
    else:
        ref_mid = 72  # middle C area

    bass_root = _clamp_bass(36 + key_root)

    # Build a 3-note treble chord: 5th below ref_mid, root near ref_mid, root+oct
    fifth_pc   = (key_root + tonic_ivs[-1]) % 12          # top interval of triad
    third_pc   = (key_root + tonic_ivs[1])  % 12

    p_fifth  = _clamp_treble(bass_split + (fifth_pc - bass_split % 12) % 12, bass_split)
    p_root   = _clamp_treble(bass_split + (key_root  - bass_split % 12) % 12, bass_split)
    p_root_h = _clamp_treble(p_root + 12)

    TREBLE_HI = 83  # keep final chord settled; p_root_h = p_root+12 <= 83+12=95... cap below
    # Nudge each toward ref_mid but don't exceed a comfortable ceiling
    for _ in range(4):
        if p_fifth < ref_mid - 12 and p_fifth + 12 <= TREBLE_HI: p_fifth += 12
        if p_root  < ref_mid - 6  and p_root  + 12 <= TREBLE_HI: p_root  += 12
    p_root_h = p_root + 12  # one octave higher — guaranteed <= TREBLE_HI + 12 = 95... clamp
    if p_root_h > 91: p_root_h = p_root  # drop high octave if it overshoots

    # Ensure ordering: fifth < root < root_high
    notes_treble = sorted({p_fifth, p_root, p_root_h})

    # Velocity: moderate — the final chord should feel settled, not loud
    vel_final = 4

    final_bar = [(0, bass_root, 16, max(2, vel_final - 1))]
    for p in notes_treble:
        final_bar.append((0, p, 16, vel_final))

    result[n - 1] = final_bar
    return result


# ── Expressive dynamics (melodic contour + agogic) ────────────────────
def _apply_expressive_dynamics(bar_events_list, phrase_bars=4,
                                section_labels=None, bass_split=58):
    """
    Shapes per-note velocities within each bar:
    1. Contour  — higher treble notes get more velocity (voice-leading tendency)
    2. Agogic   — longer notes get +1 velocity bin (natural emphasis)

    Phrase arc is applied separately in _apply_phrase_arc, AFTER
    velocity_differentiation, so the arc isn't undone by the melody boost.
    """
    result = [list(bar) for bar in bar_events_list]

    for i, bar in enumerate(result):
        if not bar:
            continue
        treble = [(j, pos, p, d, v) for j, (pos, p, d, v) in enumerate(bar) if p >= bass_split]
        if not treble:
            continue
        pitches = [p for _, _, p, _, _ in treble]
        lo, hi   = min(pitches), max(pitches)
        span     = hi - lo if hi > lo else 1
        new_bar  = list(bar)
        for j, pos, p, d, v in treble:
            frac          = (p - lo) / span
            contour_boost = int((frac - 0.33) * 9)    # −3 at bottom, +6 at top
            agogic_boost  = 1 if d >= 4 else 0
            new_v         = max(0, min(N_VEL_BINS - 1, v + contour_boost + agogic_boost))
            new_bar[j]    = (pos, p, d, new_v)
        result[i] = new_bar

    return result


def _apply_phrase_arc(bar_events_list, phrase_bars=4, section_labels=None):
    """
    Applies a soft-start / swell / taper envelope across each phrase.
    Runs AFTER velocity_differentiation so the arc isn't undone by melody boost.
    Range: 0.72 (phrase start/end) → 1.0 (phrase peak at ~65%).
    """
    result = [list(bar) for bar in bar_events_list]
    n = len(result)
    phrase_start = 0
    while phrase_start < n:
        phrase_end = min(phrase_start + phrase_bars, n)
        if section_labels is not None:
            for j in range(phrase_start + 1, phrase_end):
                if section_labels[j] != section_labels[phrase_start]:
                    phrase_end = j
                    break
        plen = phrase_end - phrase_start
        for i in range(phrase_start, phrase_end):
            t = (i - phrase_start) / max(plen - 1, 1)
            phase = math.sin(t * math.pi * 0.95)
            envelope = 0.72 + 0.28 * phase   # wider swing: 0.72→1.0 (was 0.80→1.0)
            result[i] = [
                (pos, p, d, max(0, min(N_VEL_BINS - 1, int(v * envelope + 0.5))))
                for pos, p, d, v in result[i]
            ]
        phrase_start = phrase_end
    return result


# ── MIDI humanization (timing jitter + rubato) ────────────────────────
def _humanize_midi(mid, bpm, phrase_bars=4, jitter_ms=11.0, rubato_strength=0.04):
    """
    Adds micro-timing variation to an assembled mido MidiFile:
    - Per-note Gaussian jitter (less on strong beats — pianists are more precise there)
    - Phrase-end rallentando: notes arrive slightly later in the last 25% of each phrase
    Note_on and its matching note_off are shifted together so duration is preserved.
    """
    tpb        = mid.ticks_per_beat
    tempo_us   = mido.bpm2tempo(bpm)
    ticks_per_ms = tpb / (tempo_us / 1000)
    jitter_ticks = jitter_ms * ticks_per_ms

    t16         = tpb // 4                  # ticks per 16th note
    bar_ticks   = 16 * t16                  # ticks per bar (4/4)
    phrase_ticks = phrase_bars * bar_ticks

    # Convert to absolute ticks
    track = mid.tracks[0]
    abs_events = []
    t = 0
    for msg in track:
        t += msg.time
        abs_events.append([t, msg.copy(time=0)])

    new_times = [t for t, _ in abs_events]

    # Pre-compute one offset per unique onset tick so all notes in a chord
    # shift together — independent per-note jitter makes chords sound "off pitch"
    # because the brief window of incomplete harmony reads as a wrong note.
    #
    # Only note_on events are shifted; note_off events stay on the original grid.
    # Shifting both caused same-pitch note_on/note_off reordering that produced
    # stuck notes and phantom pitches ("out of tune" artefacts).
    tick_offsets = {}
    for t_abs, msg in abs_events:
        if msg.type == 'note_on' and msg.velocity > 0 and t_abs not in tick_offsets:
            phrase_frac = (t_abs % phrase_ticks) / phrase_ticks

            # Rallentando: last 25% of phrase → notes pushed slightly later
            if phrase_frac > 0.75:
                slow_frac     = (phrase_frac - 0.75) / 0.25
                rubato_offset = int(rubato_strength * phrase_ticks * slow_frac * 0.12)
            else:
                rubato_offset = 0

            # Scale jitter by metrical weight so fast 16th-note runs stay
            # clean — independent per-tick jitter on dense passages makes
            # notes sound displaced relative to each other.
            pos_in_beat = t_abs % tpb
            if pos_in_beat == 0:
                weight = 1.0        # quarter-note beat — full jitter
            elif pos_in_beat % (tpb // 2) == 0:
                weight = 0.5        # 8th-note — moderate jitter
            else:
                weight = 0.2        # 16th-note subdivision — nearly metronomic

            rand_jitter = int(random.gauss(0, jitter_ticks * weight))
            tick_offsets[t_abs] = rubato_offset + rand_jitter

    # Apply offsets to note_ons only, clamping so no note_on lands before
    # the previous note_off on the same pitch (which would cause the note_off
    # to immediately cut the new note, producing clicks or phantom pitches).
    last_noteoff = {}  # pitch -> latest note_off tick seen in original order
    for idx, (t_abs, msg) in enumerate(abs_events):
        if msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            last_noteoff[msg.note] = max(last_noteoff.get(msg.note, 0), t_abs)
        elif msg.type == 'note_on' and msg.velocity > 0:
            offset         = tick_offsets.get(t_abs, 0)
            earliest       = last_noteoff.get(msg.note, 0)
            new_times[idx] = max(earliest, t_abs + offset)

    # Re-sort (note_off before note_on at same tick)
    order = sorted(range(len(abs_events)),
                   key=lambda i: (
                       new_times[i],
                       0 if (abs_events[i][1].type == 'note_off' or
                             (abs_events[i][1].type == 'note_on' and
                              abs_events[i][1].velocity == 0))
                       else 1
                   ))

    new_track = mido.MidiTrack()
    prev = 0
    for i in order:
        delta = max(0, new_times[i] - prev)
        new_track.append(abs_events[i][1].copy(time=delta))
        prev = new_times[i]

    new_mid = mido.MidiFile(ticks_per_beat=tpb)
    new_mid.tracks.append(new_track)
    return new_mid


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',      default='generated.mid')
    parser.add_argument('--n_bars',      type=int,   default=32)
    parser.add_argument('--chord_file',  type=str,   default=None,
                        help='JSON file with a hand-crafted chord list: [[root,qual], …]. '
                             'Bypasses the chord model. root=0-11 (C…B), '
                             'qual=0 maj,1 min,2 dim,3 aug,4 dom7,5 maj7,6 min7.')
    parser.add_argument('--seed_midi',   type=str,   default=None,
                        help='MIDI file whose bars are fed into the BarEncoder as initial '
                             'cross-attention context before generation starts.')
    parser.add_argument('--seed_bars',   type=int,   default=8,
                        help='How many bars to take from the seed MIDI (default 8, max 16).')
    parser.add_argument('--bpm',         type=int,   default=90)
    parser.add_argument('--chord_temp',  type=float, default=1.1)
    parser.add_argument('--note_temp',   type=float, default=1.0)
    parser.add_argument('--top_k',       type=int,   default=40)
    parser.add_argument('--top_p',       type=float, default=0.9)
    parser.add_argument('--max_repeat',  type=int,   default=3)
    parser.add_argument('--min_notes',       type=int,   default=4)
    parser.add_argument('--min_chord_notes', type=int,   default=2)
    parser.add_argument('--min_positions',   type=int,   default=4)
    parser.add_argument('--max_chord_notes', type=int,   default=5)
    parser.add_argument('--dur_bias',         type=float, default=0.0)
    parser.add_argument('--key_strength',    type=float, default=0.0)
    parser.add_argument('--register_strength', type=float, default=0.0)
    parser.add_argument('--register_sigma',  type=float, default=12.0)
    parser.add_argument('--form', default='none',
                        choices=['none', 'aba', 'binary', 'rondo', 'arch', 'variation'],
                        help='Musical form: none/aba/binary/rondo(ABACABA)/arch(ABCBA)/variation')
    # ── Sectional contrast ────────────────────────────────────────────
    parser.add_argument('--b_note_temp',       type=float, default=0.0,
                        help='Note temperature for B/C sections (0 = same as A)')
    parser.add_argument('--b_key_strength',    type=float, default=-1.0,
                        help='Key strength for B/C sections (-1 = same as A, lower = more chromatic)')
    parser.add_argument('--b_register_offset', type=float, default=0.0,
                        help='Semitone shift applied to register center in B/C sections')
    # ── Register arc ──────────────────────────────────────────────────
    parser.add_argument('--register_arc', type=float, default=0.0,
                        help='Total semitone shift in register from bar 0 to last bar (+ rises, − falls)')
    # ── Dynamic arc ───────────────────────────────────────────────────
    parser.add_argument('--vel_arc', type=float, default=0.0,
                        help='Sinusoidal velocity arc strength: quiet at edges, louder at midpoint')
    parser.add_argument('--motif_strength', type=float, default=0.0,
                        help='Pitch-class coherence bias for A-section returns (0 = off)')
    # ── Melodic continuity ────────────────────────────────────────────
    parser.add_argument('--melody_strength', type=float, default=0.0,
                        help='Top-voice continuity bias — pulls next bar toward previous bar\'s highest pitch')
    parser.add_argument('--melody_sigma',    type=float, default=2.5,
                        help='Gaussian width in semitones for top-voice continuity (smaller = stricter stepwise motion)')
    # ── Metric hierarchy ──────────────────────────────────────────────
    parser.add_argument('--metric_strength', type=float, default=0.0,
                        help='Logit boost on beat-1/3 position tokens (0 = off, creates flat beat distribution)')
    parser.add_argument('--metric_penalty',  type=float, default=0.0,
                        help='Logit penalty on 16th-note off-beat positions (negative number)')
    # ── Unison repulsion ──────────────────────────────────────────────
    parser.add_argument('--unison_penalty', type=float, default=0.0,
                        help='Logit penalty subtracted from top-voice pitch to discourage melody stalling on one note')
    parser.add_argument('--pitch_repeat_penalty', type=float, default=1.5,
                        help='Logit penalty applied to pitches that appeared at the previous position, discouraging re-onset of the same pitch at consecutive positions')
    parser.add_argument('--dur_repeat_penalty', type=float, default=0.8,
                        help='Logit penalty on duration tokens that were recently used, encouraging rhythmic variety within each bar')
    # ── Beat-1 bass bias ──────────────────────────────────────────────
    parser.add_argument('--bass_beat_strength', type=float, default=0.0,
                        help='Logit strength of low-register pull applied on strong beats (beats 1 and 3)')
    parser.add_argument('--bass_beat_center',   type=float, default=45.0,
                        help='MIDI pitch center for beat-1 bass bias (45=A2, 48=C3)')
    parser.add_argument('--bass_beat_sigma',    type=float, default=7.0,
                        help='Gaussian width in semitones for the beat-1 bass register bias')
    # ── Key signature ─────────────────────────────────────────────────
    parser.add_argument('--key', default='random',
                        help='Key for chord generation: C/G/D/F/Bb/Eb/Ab/etc., "random", or "none"')
    parser.add_argument('--chord_key_strength', type=float, default=0.0,
                        help='Soft logit penalty applied to non-diatonic chord roots/qualities (higher = more strictly in-key)')
    parser.add_argument('--chord_tone_strength', type=float, default=0.0,
                        help='Per-bar logit boost for notes belonging to the current chord tones (root/3rd/5th/7th). '
                             'Stacks on top of key_strength so chord tones > diatonic passing tones > chromatic.')
    # ── Phrase arc ────────────────────────────────────────────────────
    parser.add_argument('--phrase_arc',  type=float, default=0.0,
                        help='Register arch amplitude in semitones within each phrase (0 = off)')
    parser.add_argument('--phrase_bars', type=int,   default=4,
                        help='Phrase length in bars for the phrase arc cycle')
    # ── Left-hand Alberti bass ────────────────────────────────────────
    parser.add_argument('--lh_pattern', default='none',
                        choices=['procedural', 'alberti', 'stride', 'murky', 'waltz', 'walking', 'none'],
                        help='Left-hand pattern imposed below bass_split MIDI pitch')
    parser.add_argument('--lh_bass_split', type=int, default=58,
                        help='MIDI pitch below which model notes are replaced by LH pattern')
    parser.add_argument('--debug_stages', action='store_true',
                        help='Save intermediate MIDIs at each post-processing stage')
    parser.add_argument('--raw', action='store_true',
                        help='Skip all post-processing heuristics; output the raw model generation')
    parser.add_argument('--lh_vel', type=int, default=3,
                        help='Velocity bin (0-7) for LH Alberti root; inner notes get lh_vel-1')
    # ── Long-range coherence ──────────────────────────────────────────
    parser.add_argument('--coherence_decay', type=float, default=0.0,
                        help='Progressive temperature reduction over the piece (0=off). '
                             'bar_temp is scaled by max(1-decay*progress, 0.75) so later '
                             'bars are more focused and less likely to drift.')
    # ── Velocity differentiation ──────────────────────────────────────
    parser.add_argument('--melody_boost',   type=int, default=1,
                        help='Velocity bins added to melody (highest) note at each position')
    parser.add_argument('--accomp_reduce',  type=int, default=1,
                        help='Velocity bins subtracted from non-melody notes at each position')
    # ── Humanization ──────────────────────────────────────────────────
    parser.add_argument('--no_humanize', action='store_true',
                        help='Skip timing humanization and expressive dynamics (outputs a metronomic MIDI)')
    parser.add_argument('--jitter_ms',   type=float, default=11.0,
                        help='Timing jitter std-dev in milliseconds (default 11 ms)')
    # ── Cadence enforcement ───────────────────────────────────────────
    parser.add_argument('--enforce_cadence', action='store_true', default=True,
                        help='Resolve the final cadence; mid-phrase cadences stay model-driven')
    parser.add_argument('--no_enforce_cadence', dest='enforce_cadence', action='store_false')
    # ── Phrase melodic goal ───────────────────────────────────────────
    parser.add_argument('--phrase_goal_range', type=float, default=0.0,
                        help='Semitones above phrase-start top voice to target at phrase midpoint (0 = off)')
    # ── Phrase repetition ─────────────────────────────────────────────
    parser.add_argument('--phrase_repeat_prob', type=float, default=0.5,
                        help='Probability that a phrase is repeated verbatim (0 = never, 1 = always)')
    # ── Bass continuity ───────────────────────────────────────────────
    parser.add_argument('--bass_strength', type=float, default=0.0,
                        help='Bottom-voice continuity bias strength (0 = off)')
    parser.add_argument('--bass_sigma',    type=float, default=5.0,
                        help='Gaussian width in semitones for bass-voice continuity')
    # ── Cadential endings ─────────────────────────────────────────────
    parser.add_argument('--cadence_notes', type=int,   default=2,
                        help='max_chord_notes cap for the last bar of each phrase (cadential thinning)')
    parser.add_argument('--cadence_dur_scale', type=float, default=2.0,
                        help='dur_bias multiplier for the cadential bar')
    parser.add_argument('--no_bias', action='store_true',
                        help='Zero ALL biases including key_strength and chord_tone_strength — fully trust the model')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    if args.no_bias:
        args.key_strength        = 0.0
        args.chord_tone_strength = 0.0
        args.chord_key_strength  = 0.0

    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))
    print(f"Device: {device}")

    print("Loading chord model…")
    chord_model = load_model('checkpoints/chord_model.pt', device)
    note_model  = _load_note_model(device)
    bar_encoder = _load_bar_encoder(device)

    # ── Stage 1: chord progression ────────────────────────────────────
    # Resolve key signature
    import random as _random
    _KEY_NAMES = {'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,
                  'F':5,'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11}
    key_root = None
    if args.key == 'none':
        diatonic_roots    = None
        key_display       = 'none (chromatic)'
    elif args.key == 'random':
        key_root          = _random.randint(0, 11)
        diatonic_roots    = diatonic_roots_for_key(key_root)
        key_display       = f'{ROOTS[key_root]} major (random)'
    else:
        key_root = _KEY_NAMES.get(args.key)
        if key_root is None:
            raise ValueError(f"Unknown key: {args.key!r}. Use C/G/D/F/Bb/Eb/Ab/etc.")
        diatonic_roots = diatonic_roots_for_key(key_root)
        key_display    = f'{ROOTS[key_root]} major'
    print(f"Key: {key_display}")

    if args.chord_file:
        import json as _json
        with open(args.chord_file) as _f:
            _raw = _json.load(_f)
        chords = [tuple(c) for c in _raw]
        # Pad or trim to n_bars
        while len(chords) < args.n_bars:
            chords.append(chords[-1])
        chords = chords[:args.n_bars]
        print(f"\nUsing hand-crafted chords from {args.chord_file} ({len(chords)} bars)")
    else:
        print(f"\nGenerating {args.n_bars}-bar chord progression…")
        chords = generate_chords(chord_model, args.n_bars,
                                 args.chord_temp, args.top_k, args.top_p, device,
                                 max_repeat=args.max_repeat,
                                 diatonic_roots=diatonic_roots,
                                 chord_key_strength=args.chord_key_strength,
                                 key_root=key_root)

        last_real = next((c for c in reversed(chords) if c is not None), None)
        while len(chords) < args.n_bars:
            chords.append(last_real)
        chords = chords[:args.n_bars]
        chords = _postprocess_chords(chords, max_run=args.max_repeat)
    chords = _resolve_none_chords(chords, key_root)

    # Apply form: reorder chords and get per-bar section labels
    chords, section_labels = _apply_form(chords, args.n_bars, args.form)
    chords = _resolve_none_chords(chords, key_root)
    is_variation = (args.form == 'variation')

    # Cadence enforcement: override phrase-final bars to V7→I in A sections
    if args.enforce_cadence and diatonic_roots is not None and args.phrase_bars > 1:
        chords = _enforce_cadences(
            chords, section_labels, args.phrase_bars, key_root,
            tonic_qual_override=0 if key_root is not None else None)
        chords = _resolve_none_chords(chords, key_root)
        print("Cadence enforcement: final resolution only; mid-phrase cadences stay model-driven")

    form_name = {'none':'through-composed','aba':'ABA','binary':'Binary (AB)',
                 'rondo':'Rondo (ABACABA)','arch':'Arch (ABCBA)',
                 'variation':'Theme & Variations'}.get(args.form, args.form)
    print(f"Form: {form_name}")
    print("Chord progression:")
    for i, c in enumerate(chords):
        label = f"{ROOTS[c[0]]} {QUALITIES[c[1]]}" if c else "(none)"
        print(f"  bar {i+1:3d} [{section_labels[i]}]: {label}")

    # ── Precompute logit biases ───────────────────────────────────────
    dur_bias_tensor = build_dur_bias(args.dur_bias, device) if args.dur_bias > 0 else None

    # Metric hierarchy: beat-aligned position preference
    metric_bias = build_metric_bias(args.metric_strength, args.metric_penalty, device) \
                  if args.metric_strength != 0 else None

    # Beat-1 bass bias: low-register pull applied on strong beats inside generate_bar
    beat_bass_bias = register_bias(args.bass_beat_center, args.bass_beat_sigma,
                                   args.bass_beat_strength, device) \
                     if args.bass_beat_strength > 0 else None

    # Key bias for A sections
    key_bias_A = build_key_bias(chords, device, strength=_scaled_strength(args.key_strength),
                                tonic_root=key_root, mode='major' if key_root is not None else None) \
                 if args.key_strength > 0 else None
    # Key bias for B/C sections (may differ in strength)
    b_ks = args.b_key_strength if args.b_key_strength >= 0 else args.key_strength
    key_bias_B = build_key_bias(chords, device, strength=_scaled_strength(b_ks),
                                tonic_root=key_root, mode='major' if key_root is not None else None) \
                 if b_ks > 0 else None

    if key_bias_A is not None:
        if key_root is None:
            tonic_root = max(
                {c[0] for c in chords if c is not None},
                key=lambda r: sum(1 for c in chords if c is not None and c[0] == r)
            )
            print(f"Tonal center inferred: {ROOTS[tonic_root]}")
        else:
            print(f"Tonal center: {ROOTS[key_root]} major")

    # ── Stage 2: notes bar by bar ─────────────────────────────────────
    print(f"\nFilling {len(chords)} bars with notes…")
    bar_events_list = []
    empty      = 0
    reg_center = None

    # Motif heuristic state
    MOTIF_BARS    = 2       # collect pitch classes from first N bars of first A section
    ANCHOR_BARS   = 4       # opening bars pinned permanently in BarEncoder memory (≤ MAX_HISTORY)
    motif_pitches = []      # raw pitches gathered during first A block
    motif_bias    = None    # built once we leave the first A section
    first_a_done  = False   # True after first A block ends
    a_bars_seen   = 0       # A bars counted within first A block

    # Melodic / bass continuity state
    top_voice    = None     # highest pitch of the previous bar
    bottom_voice = None     # lowest pitch of the previous bar

    # Cross-bar model memory: raw note tokens from the previous bar fed as
    # prefix context so the note model sees what was actually played.
    # Reset at section boundaries (A→B, B→A) so contrasting sections start fresh.
    prev_bar_tokens  = None
    prev_bar_section = None
    bar_history      = []   # completed bar token lists for cross-attention
    anchor_history   = []   # first MOTIF_BARS bars — always pinned at front of memory

    # Seed BarEncoder memory from an external MIDI file so the model generates
    # with that piece's style already in its cross-attention context.
    if args.seed_midi:
        try:
            _, seed_bar_samples = midi_to_training_data(args.seed_midi)
            n_seed = min(args.seed_bars, 16, len(seed_bar_samples))
            bar_history = [list(notes) for _, notes in seed_bar_samples[:n_seed]]
            print(f"Seeding with {n_seed} bars from {args.seed_midi}")
        except Exception as _e:
            print(f"Warning: could not load seed MIDI ({_e}); continuing without seed")

    # Phrase melodic goal state
    import random as _rand
    phrase_start_pitch = None   # top voice at phrase start
    phrase_goal_pitch  = None   # target peak at phrase midpoint

    for i, curr in enumerate(chords):
        section       = section_labels[i]
        var_dur_scale = 1.0   # overridden in variation B/C sections

        # Per-section generation parameters
        if section == 'A' or args.form in ('none',):
            bar_temp      = args.note_temp
            key_bias      = key_bias_A
            reg_offset    = 0.0
            bar_min_pos   = args.min_positions
            bar_max_chord = args.max_chord_notes
        elif is_variation:
            # Variation: each repetition escalates in density, temperature, and note length
            var_step      = {'A': 0, 'B': 1, 'C': 2}.get(section, 0)
            bar_temp      = args.note_temp + var_step * 0.08
            key_bias      = key_bias_A
            reg_offset    = 0.0
            bar_min_pos   = min(args.min_positions + var_step, 6)
            bar_max_chord = min(args.max_chord_notes + var_step, 6)
            var_dur_scale = [1.0, 1.3, 1.7][var_step]
        else:
            # B or C section: apply contrast params
            bar_temp      = args.b_note_temp if args.b_note_temp > 0 else args.note_temp
            key_bias      = key_bias_B
            reg_offset    = args.b_register_offset
            bar_min_pos   = args.min_positions
            bar_max_chord = args.max_chord_notes

        # Cadential bar: last bar of each phrase gets thinned texture and longer notes
        is_cadence = args.phrase_bars > 1 and (i % args.phrase_bars == args.phrase_bars - 1)
        if is_cadence:
            bar_max_chord  = max(args.min_chord_notes, min(bar_max_chord, args.cadence_notes))
            var_dur_scale *= args.cadence_dur_scale

        # Coherence annealing: gently cool bar_temp over the piece so later bars
        # make more confident, less drifting predictions.  Early bars explore freely;
        # later bars converge toward consistent patterns without sounding mechanical.
        if args.coherence_decay > 0 and args.n_bars > 1:
            t_frac   = i / (args.n_bars - 1)            # 0.0 (bar 0) → 1.0 (last bar)
            bar_temp = bar_temp * max(1.0 - args.coherence_decay * t_frac, 0.75)

        # Linear register arc (additive on top of section offset)
        if args.register_arc != 0.0:
            arc_frac   = i / max(args.n_bars - 1, 1)
            reg_offset += args.register_arc * arc_frac

        # Phrase arc: sinusoidal register arch within each phrase cycle
        # Gives each phrase a rise-and-fall shape rather than a flat register plateau.
        if args.phrase_arc != 0.0 and args.phrase_bars > 1:
            phrase_pos  = i % args.phrase_bars
            reg_offset += args.phrase_arc * math.sin(math.pi * phrase_pos / args.phrase_bars)

        # Assemble extra_bias for this bar
        base_dur = dur_bias_tensor * var_dur_scale if (dur_bias_tensor is not None and var_dur_scale != 1.0) \
                   else (dur_bias_tensor.clone() if dur_bias_tensor is not None else None)
        extra_bias = base_dur
        if key_bias is not None:
            extra_bias = key_bias.clone() if extra_bias is None else extra_bias + key_bias
        # Per-bar chord-tone bias: boosts root/3rd/5th/7th of the current chord so
        # diatonic non-chord tones are less likely to clash against the harmony.
        if args.chord_tone_strength > 0:
            ct_bias = build_chord_tone_bias(curr, device, _scaled_strength(args.chord_tone_strength))
            if ct_bias is not None:
                extra_bias = ct_bias if extra_bias is None else extra_bias + ct_bias
        if metric_bias is not None:
            extra_bias = metric_bias.clone() if extra_bias is None else extra_bias + metric_bias

        # Register pull: toward (reg_center + reg_offset) if we have a previous bar,
        # otherwise seed from middle-C + offset so the arc starts somewhere sensible.
        if args.register_strength > 0:
            rc = (reg_center + reg_offset) if reg_center is not None else (60.0 + reg_offset)
            rb = register_bias(rc, args.register_sigma, args.register_strength, device)
            extra_bias = rb if extra_bias is None else extra_bias + rb

        # Top-voice continuity with phrase melodic goal.
        # At phrase start: save top_voice and sample an upward goal pitch.
        # First half of phrase: interpolate melody target upward to the goal.
        # Second half: interpolate back downward toward phrase-start pitch.
        # This creates a rise-and-fall contour rather than aimless wandering.
        if args.melody_strength > 0 and top_voice is not None:
            phrase_pos = i % args.phrase_bars if args.phrase_bars > 1 else 0
            if phrase_pos == 0:
                phrase_start_pitch = top_voice
                if args.phrase_goal_range > 0:
                    phrase_goal_pitch = top_voice + _rand.uniform(
                        args.phrase_goal_range * 0.4, args.phrase_goal_range)
                else:
                    phrase_goal_pitch = top_voice

            if phrase_goal_pitch is not None and phrase_start_pitch is not None \
                    and args.phrase_bars > 1:
                frac = phrase_pos / max(args.phrase_bars - 1, 1)   # 0.0 → 1.0
                if frac <= 0.5:
                    t = frac * 2                                    # 0→1 over first half
                    melody_target = phrase_start_pitch + (phrase_goal_pitch - phrase_start_pitch) * t
                else:
                    t = (frac - 0.5) * 2                           # 0→1 over second half
                    melody_target = phrase_goal_pitch - (phrase_goal_pitch - phrase_start_pitch) * t
            else:
                melody_target = float(top_voice)

            mv = register_bias(melody_target, args.melody_sigma, args.melody_strength, device)
            # Unison repulsion: subtract a small penalty at the exact top-voice pitch
            # so the melody moves by at least a semitone rather than hovering.
            if args.unison_penalty > 0:
                ui = NOTE_ON_OFF + (top_voice - _MIN_PITCH)
                if NOTE_ON_OFF <= ui < _NOTE_ON_END:
                    mv[ui] -= args.unison_penalty
            extra_bias = mv if extra_bias is None else extra_bias + mv

        # Bass-voice continuity: pull toward the previous bar's lowest pitch.
        # Anchors the bass line so it moves purposefully rather than leaping randomly.
        if args.bass_strength > 0 and bottom_voice is not None:
            bv = register_bias(bottom_voice, args.bass_sigma, args.bass_strength, device)
            extra_bias = bv if extra_bias is None else extra_bias + bv

        # Velocity arc
        if args.vel_arc > 0:
            vb = vel_arc_bias(i, args.n_bars, args.vel_arc, device)
            extra_bias = vb if extra_bias is None else extra_bias + vb

        # Cross-bar pitch repetition penalty: penalize pitches that were dominant
        # in the previous bar to break autoregressive feedback loops.
        if bar_history and args.pitch_repeat_penalty > 0:
            prev_tokens = bar_history[-1]
            pitch_counts: dict = {}
            for tok in prev_tokens:
                if NOTE_ON_OFF <= tok < NOTE_ON_OFF + 88:
                    midi_p = tok - NOTE_ON_OFF + 21
                    pitch_counts[midi_p] = pitch_counts.get(midi_p, 0) + 1
            if pitch_counts:
                total_prev = sum(pitch_counts.values())
                cb_bias = torch.zeros(NOTE_VOCAB, device=device)
                for midi_p, cnt in pitch_counts.items():
                    dominance = cnt / total_prev
                    if dominance > 0.12:   # only penalize pitches with >12% share
                        penalty = args.pitch_repeat_penalty * dominance * 1.2
                        tok_idx = NOTE_ON_OFF + (midi_p - 21)
                        if 0 <= tok_idx < NOTE_VOCAB:
                            cb_bias[tok_idx] -= penalty
                extra_bias = cb_bias if extra_bias is None else extra_bias + cb_bias

        # Motif coherence: apply pitch-class bias on all bars once the motif is known.
        # Applying globally (not just A returns) keeps the opening fingerprint as a
        # soft anchor throughout — the primary defence against long-range drift.
        # Strength is halved in B/C sections so it doesn't undermine contrast.
        if motif_bias is not None and first_a_done:
            mb = motif_bias if (section == 'A' or is_variation) else motif_bias * 0.5
            extra_bias = mb.clone() if extra_bias is None else extra_bias + mb

        # Reset cross-bar memory at section boundaries so B-section notes don't
        # contaminate A2's conditioning in ABA/rondo/arch forms. Without this,
        # the A2 section starts sounding like a continuation of B rather than a
        # return to A — the main driver of across-piece incoherence in form pieces.
        if section != prev_bar_section:
            # Clear prev_bar_tokens so the first bar of the new section doesn't
            # echo the previous section's closing phrase (removes the seam's
            # echo-chain cause without cutting the melodic thread entirely).
            # Keep bar_history intact — the BarEncoder's full accumulated memory
            # is what makes B and C feel like they belong to the same piece as A.
            # The chord progression change is sufficient to drive section contrast.
            prev_bar_tokens = None
        prev_bar_section = section

        # Phrase-boundary reset: clear prev_bar_tokens at the start of each phrase
        # to break feedback chains (a repetitive bar echoing into the next via the
        # prefix).  bar_history is kept so BarEncoder structural memory persists.
        # Section boundaries already do a full reset; this is a lighter within-section
        # safeguard that fires every phrase_bars bars regardless of form.
        if i > 0 and args.phrase_bars > 1 and (i % args.phrase_bars == 0):
            prev_bar_tokens = None
        # Bar-1 reset: bar 0's strong pattern can echo into bar 1 via prev_bar_tokens,
        # making the opening two bars identical. Clearing at i=1 breaks this chain
        # without affecting the BarEncoder's structural memory.
        if i == 1:
            prev_bar_tokens = None

        prev   = chords[i - 1] if i > 0 else None
        prefix = chord_prefix(prev, curr, prev_bar_tokens)
        # Build effective history: anchor bars (opening identity) + recent bars.
        # MAX_HISTORY=16; anchor_history holds the first MOTIF_BARS bars so the
        # model never forgets where the piece started, even at bar 40+.
        if anchor_history:
            n_recent    = min(16 - len(anchor_history), len(bar_history))
            eff_history = anchor_history + bar_history[-n_recent:] if n_recent > 0 else anchor_history
        else:
            eff_history = bar_history
        memory, mem_key_mask = build_inference_memory(bar_encoder, eff_history, device)
        events, raw_tokens = generate_bar(note_model, prefix,
                                          bar_temp, args.top_k, args.top_p, device,
                                          min_notes=args.min_notes,
                                          min_chord_notes=args.min_chord_notes,
                                          min_positions=bar_min_pos,
                                          max_chord_notes=bar_max_chord,
                                          extra_bias=extra_bias.unsqueeze(0) if extra_bias is not None else None,
                                          beat_bass_bias=beat_bass_bias,
                                          pitch_repeat_penalty=args.pitch_repeat_penalty,
                                          dur_repeat_penalty=args.dur_repeat_penalty,
                                          memory=memory,
                                          memory_key_mask=mem_key_mask)
        prev_bar_tokens = raw_tokens   # feed into next bar's prefix
        bar_history.append(raw_tokens)
        # Pin the opening bars as permanent anchors once the motif window fills.
        # These are always prepended to the effective history so the BarEncoder
        # never loses sight of the piece's opening identity, even at bar 40+.
        if not anchor_history and len(bar_history) >= ANCHOR_BARS:
            anchor_history = list(bar_history[:ANCHOR_BARS])
        bar_events_list.append(events)
        if not events:
            empty += 1
        else:
            pitches      = [e[1] for e in events]
            reg_center   = sum(pitches) / len(pitches)
            top_voice    = max(pitches)
            bottom_voice = min(pitches)

        # Motif: collect from first A block, finalize when leaving it.
        # For through-composed pieces (form='none', all bars labelled 'A'),
        # finalize after MOTIF_BARS bars because we never reach a non-A section.
        if not first_a_done:
            if section == 'A':
                if a_bars_seen < MOTIF_BARS and events:
                    motif_pitches.extend(e[1] for e in events)
                a_bars_seen += 1
            finalize_motif = (section != 'A') or (a_bars_seen >= MOTIF_BARS)
            if finalize_motif:
                first_a_done = True
                if motif_pitches and args.motif_strength > 0:
                    # Filter to in-key pitch classes only: chromatic accidents in the first
                    # bars (model errors) must not be permanently reinforced on A returns.
                    if diatonic_roots is not None:
                        motif_pitches = [p for p in motif_pitches if p % 12 in diatonic_roots]
                    motif_bias = build_motif_bias(motif_pitches, device, args.motif_strength) if motif_pitches else None
                    pcs = sorted({p % 12 for p in motif_pitches})
                    print(f"Motif: {len(motif_pitches)} notes, pitch classes {pcs} (in-key only)")

    if not args.raw:
        # ── Phrase repetition ─────────────────────────────────────────────
        if args.phrase_repeat_prob > 0 and args.phrase_bars > 1:
            bar_events_list = _apply_phrase_repetition(
                bar_events_list, section_labels, args.phrase_bars, args.phrase_repeat_prob)
            # ABAB: repetitions are 2 phrases apart, so check offset = 2 * phrase_bars
            skip = args.phrase_bars * 2
            repeated = sum(1 for i in range(skip, len(bar_events_list))
                           if bar_events_list[i] == bar_events_list[i - skip])
            print(f"Phrase repetition (ABAB): ~{repeated} bars match phrase 2 slots earlier")

        # ── ABA recapitulation ────────────────────────────────────────────
        # Copy A1 note content into A2 (and any other repeated section runs)
        # so the listener hears the same theme returning rather than fresh
        # unrelated material — giving the piece its communicative narrative arc.
        if args.form in ('aba', 'arch'):
            bar_events_list = _apply_aba_recapitulation(bar_events_list, section_labels,
                                                        skip_start=args.phrase_bars)

        # ── Left-hand bass ────────────────────────────────────────────────
        if args.lh_pattern == 'procedural':
            bar_events_list = _apply_procedural_bass(
                bar_events_list, chords,
                bass_split=args.lh_bass_split,
                lh_vel=args.lh_vel,
                phrase_bars=args.phrase_bars,
                key_root=key_root,
            )
            print("LH pattern: procedural bass generated")
        elif args.lh_pattern != 'none':
            bar_events_list = [
                _impose_lh_pattern(events, chords[i],
                                   bass_split=args.lh_bass_split,
                                   lh_vel=args.lh_vel,
                                   pattern=args.lh_pattern,
                                   bar_idx=i,
                                   next_chord=chords[i + 1] if i + 1 < len(chords) else None)
                for i, events in enumerate(bar_events_list)
            ]
            print(f"LH pattern: {args.lh_pattern} bass imposed")

        bar_events_list = _ensure_treble_presence(
            bar_events_list, chords,
            bass_split=args.lh_bass_split,
            phrase_bars=args.phrase_bars,
            section_labels=section_labels)

        # ── Truncate treble notes that sustain into a new chord as non-chord-tones ──
        bar_events_list = _truncate_cross_chord_sustains(
            bar_events_list, chords, bass_split=args.lh_bass_split)

        # ── Thin treble texture ───────────────────────────────────────────
        bar_events_list = _thin_treble_texture(
            bar_events_list, bass_split=args.lh_bass_split)
        if getattr(args, 'debug_stages', False):
            bars_to_midi(bar_events_list, bpm=args.bpm).save(args.output + '.stage1_thin.mid')

        # ── Polish: collapse repetitive runs in all voices ────────────────
        bar_events_list = _polish_top_voice(
            bar_events_list, bass_split=args.lh_bass_split)
        bar_events_list = _collapse_bass_runs(
            bar_events_list, bass_split=args.lh_bass_split)
        bar_events_list = _collapse_inner_voice_runs(
            bar_events_list, bass_split=args.lh_bass_split)

        if getattr(args, 'debug_stages', False):
            bars_to_midi(bar_events_list, bpm=args.bpm).save(args.output + '.stage2_polish.mid')

        # ── Melody smoothing (cross-bar leaps + tritones) ────────────────
        bar_events_list = _smooth_melody_leaps(
            bar_events_list, bass_split=args.lh_bass_split,
            section_labels=section_labels)

        # ── Intra-bar melodic smoothing (within-bar harsh intervals) ─────
        if diatonic_roots is not None:
            bar_events_list = _smooth_intra_bar_leaps(
                bar_events_list, diatonic_roots, bass_split=args.lh_bass_split)

        # ── Break cross-bar top-voice loops (after smoothing so the loop-
        #    breaker's displacements aren't reversed by the leap smoother) ──
        bar_events_list = _break_top_voice_loops(
            bar_events_list, bass_split=args.lh_bass_split)

        if getattr(args, 'debug_stages', False):
            bars_to_midi(bar_events_list, bpm=args.bpm).save(args.output + '.stage3_smooth.mid')

        # ── Cross-bar ties ────────────────────────────────────────────────
        bar_events_list = _add_cross_bar_ties(
            bar_events_list, chords, bass_split=args.lh_bass_split,
            section_labels=section_labels)

        # ── Vertical dissonance filter ────────────────────────────────────
        bar_events_list = _reduce_vertical_dissonance(
            bar_events_list, diatonic_roots or set(), chords=chords,
            bass_split=args.lh_bass_split)

        # ── Chromatic clash filter ────────────────────────────────────────
        if diatonic_roots is not None:
            bar_events_list = _soften_chromatic_clashes(
                bar_events_list, diatonic_roots, bass_split=args.lh_bass_split)

        # ── m2 safety pass (catches any remaining adjacent-semitone clashes) ─
        bar_events_list = _strip_m2_clashes(bar_events_list, bass_split=args.lh_bass_split)

        # ── Expressive dynamics (phrase arcs + melodic contour) ──────────
        if not args.no_humanize:
            bar_events_list = _apply_expressive_dynamics(
                bar_events_list, phrase_bars=args.phrase_bars,
                section_labels=section_labels, bass_split=args.lh_bass_split)

        # ── Velocity differentiation ──────────────────────────────────────
        if args.melody_boost > 0 or args.accomp_reduce > 0:
            bar_events_list = _apply_velocity_differentiation(
                bar_events_list, args.melody_boost, args.accomp_reduce)

        # ── Phrase arc (after melody boost so it isn't undone) ───────────
        if not args.no_humanize:
            bar_events_list = _apply_phrase_arc(
                bar_events_list, phrase_bars=args.phrase_bars,
                section_labels=section_labels)

        # ── Section crossfade (dynamic breath at A→B, B→A transitions) ──
        if args.form != 'none':
            bar_events_list = _apply_section_crossfade(
                bar_events_list, section_labels)

        # ── Post-climax breath ────────────────────────────────────────────
        bar_events_list = _apply_post_climax_breath(
            bar_events_list, section_labels, bass_split=args.lh_bass_split)

        # ── Final resolution (fade + tonic landing) ───────────────────────
        if key_root is not None:
            qual_counts = {}
            for c in chords:
                if c:
                    qual_counts[c[1]] = qual_counts.get(c[1], 0) + 1
            tonic_qual_detected = (
                1 if qual_counts.get(1, 0) + qual_counts.get(6, 0)
                     > qual_counts.get(0, 0) + qual_counts.get(5, 0)
                else 0
            )
            bar_events_list = _apply_final_resolution(
                bar_events_list, chords, key_root, tonic_qual_detected,
                section_labels, bass_split=args.lh_bass_split)
            # Final resolution rebuilds the last bar with new notes — re-run
            # carry truncation so any pre-existing cross-bar sustains into the
            # rebuilt bar don't create m2 clashes with the new chord.
            bar_events_list = _truncate_cross_chord_sustains(
                bar_events_list, chords, bass_split=args.lh_bass_split)
            bar_events_list = _strip_m2_clashes(
                bar_events_list, bass_split=args.lh_bass_split)

    # ── Assemble & save ───────────────────────────────────────────────
    mid = bars_to_midi(bar_events_list, bpm=args.bpm)
    if not args.no_humanize:
        mid = _humanize_midi(mid, args.bpm,
                             phrase_bars=args.phrase_bars,
                             jitter_ms=args.jitter_ms)
    mid.save(args.output)
    total = sum(len(e) for e in bar_events_list)
    print(f"\nSaved  → {args.output}")
    print(f"Notes  : {total}  |  Empty bars: {empty}/{len(chords)}")


if __name__ == '__main__':
    main()
