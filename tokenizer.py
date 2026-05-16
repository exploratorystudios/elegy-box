"""
MIDI tokenization for the chord-conditioned hierarchical model.

Two vocabularies:
  CHORD vocab  — for the chord-sequence model (tiny GPT over bar-level harmony)
  NOTE vocab   — for the note-filling model (GPT conditioned on chord prefix)

Encoding strategy (REMI-like, bar-aligned):
  Each bar is represented as:
    chord model  : [ROOT, QUAL]  (one pair per bar)
    note model   : [PREV_ROOT, PREV_QUAL, CURR_ROOT, CURR_QUAL, ...note events..., BAR_END]
  Note events within a bar: [POSITION (NOTE_ON DURATION VELOCITY)+]+ BAR_END
  POS is emitted once per position group; multiple notes can share a position (chords).
  Position is the 1/16th-note slot within the bar (0-15).
"""

import mido
from collections import defaultdict

# ──────────────────────────────────────────────
#  CHORD vocabulary  (26 tokens)
# ──────────────────────────────────────────────
CHORD_PAD      = 0
CHORD_BOS      = 1
CHORD_EOS      = 2
CHORD_ROOT_OFF = 3          # 3..14  (12 roots: C=3 … B=14)
CHORD_QUAL_OFF = 15         # 15..24 (10 qualities)
CHORD_NONE     = 25         # bar with no clear chord
CHORD_VOCAB    = 26

ROOTS     = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
QUALITIES = ['maj','min','dim','aug','dom7','maj7','min7','hdim7','sus2','sus4']

# Pitch-class offsets relative to root
CHORD_TEMPLATES = {
    'maj':   [0, 4, 7],
    'min':   [0, 3, 7],
    'dim':   [0, 3, 6],
    'aug':   [0, 4, 8],
    'dom7':  [0, 4, 7, 10],
    'maj7':  [0, 4, 7, 11],
    'min7':  [0, 3, 7, 10],
    'hdim7': [0, 3, 6, 10],
    'sus2':  [0, 2, 7],
    'sus4':  [0, 5, 7],
}

# ──────────────────────────────────────────────
#  NOTE vocabulary  (153 tokens)
# ──────────────────────────────────────────────
NOTE_PAD       = 0
NOTE_BAR_END   = 1
# chord conditioning prefix (same layout as chord vocab, offset by 2)
NOTE_ROOT_OFF  = 2          # 2..13
NOTE_QUAL_OFF  = 14         # 14..23
NOTE_NONE      = 24
NOTE_POS_OFF   = 25         # 25..40  (16 positions, 0-15)
NOTE_ON_OFF    = 41         # 41..128 (pitches 21-108, piano range)
NOTE_DUR_OFF   = 129        # 129..144 (durations 1-16 in 1/16th notes)
NOTE_VEL_OFF   = 145        # 145..152 (8 velocity bins)
NOTE_VOCAB     = 153

MIN_PITCH      = 21         # A0
MAX_PITCH      = 108        # C8
POSITIONS      = 16         # 1/16th-note slots per bar (4/4)
MAX_DUR        = 16         # max note duration in slots
N_VEL_BINS     = 8

# Maximum number of previous-bar note tokens included in the prefix.
# The note model prefix becomes:
#   [prev_root, prev_qual, <prev_bar_tokens[-MAX_PREV_BAR:]>, curr_root, curr_qual]
# This gives the model actual cross-bar memory — it sees what notes were played
# in the previous bar when generating the current bar.
MAX_PREV_BAR   = 48


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────
def _vel_bin(v):
    return min(v * N_VEL_BINS // 128, N_VEL_BINS - 1)

def _bin_vel(b):
    return max(1, b * 128 // N_VEL_BINS + 8)

def _score_chord(pc_set, root, qual):
    template = set((root + t) % 12 for t in CHORD_TEMPLATES[qual])
    hits      = len(pc_set & template)
    outside   = len(pc_set - template)
    return hits / len(template) - 0.1 * outside

def _best_chord(pitches):
    if not pitches:
        return None
    pc_set = set(p % 12 for p in pitches)
    best_score, best = -1, None
    for root in range(12):
        for qi, qual in enumerate(QUALITIES):
            s = _score_chord(pc_set, root, qual)
            if s > best_score:
                best_score, best = s, (root, qi)
    return best if best_score >= 0.4 else None


# ──────────────────────────────────────────────
#  MIDI → internal note list
# ──────────────────────────────────────────────
def _parse_midi(path):
    """Return list of (start_slot, pitch, dur_slots, vel_bin), or None on failure."""
    try:
        mid = mido.MidiFile(path)
    except Exception:
        return None

    tpb  = mid.ticks_per_beat
    t16  = max(1, tpb // 4)   # ticks per 1/16th note

    raw = []
    for track in mid.tracks:
        tick  = 0
        active = {}     # pitch -> (start_tick, velocity)
        for msg in track:
            tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                active[msg.note] = (tick, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active:
                    start, vel = active.pop(msg.note)
                    dur = tick - start
                    if dur > 0 and MIN_PITCH <= msg.note <= MAX_PITCH:
                        raw.append((start, msg.note, dur, vel))

    if not raw:
        return None

    raw.sort()
    notes = []
    for start, pitch, dur, vel in raw:
        s = round(start / t16)
        d = max(1, min(MAX_DUR, round(dur / t16)))
        notes.append((s, pitch, d, _vel_bin(vel)))
    return notes


# ──────────────────────────────────────────────
#  Group notes into bars
# ──────────────────────────────────────────────
def _notes_to_bars(notes):
    bars = defaultdict(list)
    for slot, pitch, dur, vel in notes:
        bar = slot // POSITIONS
        pos = slot % POSITIONS
        bars[bar].append((pos, pitch, dur, vel))
    return bars


# ──────────────────────────────────────────────
#  Public encoding functions
# ──────────────────────────────────────────────
def encode_chord_sequence(chords, n_bars):
    """Build token list for chord model: [BOS, ROOT, QUAL, ROOT, QUAL, …, EOS]."""
    tokens = [CHORD_BOS]
    for b in range(n_bars):
        c = chords.get(b)
        if c is None:
            tokens += [CHORD_NONE, CHORD_NONE]
        else:
            tokens += [CHORD_ROOT_OFF + c[0], CHORD_QUAL_OFF + c[1]]
    tokens.append(CHORD_EOS)
    return tokens


def encode_bar(prev_chord, curr_chord, events, prev_bar_tokens=None):
    """
    Build (prefix, note_tokens) for one bar.

    prefix       — chord-conditioning tokens + optional previous-bar context
    note_tokens  — POSITION, NOTE_ON, DURATION, VELOCITY groups + BAR_END

    With prev_bar_tokens=None (first bar or section start), the prefix is the
    original 4-token format: [prev_root, prev_qual, curr_root, curr_qual].

    With prev_bar_tokens supplied, the prefix becomes:
        [prev_root, prev_qual, <prev_bar_tokens[-MAX_PREV_BAR:]>, curr_root, curr_qual]
    giving the model cross-bar memory of what was actually played in bar N-1
    when generating bar N.  Loss mask still starts at the first note token.
    """
    def _chord_tokens(c, root_off, qual_off, none_tok):
        if c is None:
            return [none_tok, none_tok]
        return [root_off + c[0], qual_off + c[1]]

    prev_ctx = list(prev_bar_tokens[-MAX_PREV_BAR:]) if prev_bar_tokens else []
    prefix = (_chord_tokens(prev_chord, NOTE_ROOT_OFF, NOTE_QUAL_OFF, NOTE_NONE) +
              prev_ctx +
              _chord_tokens(curr_chord, NOTE_ROOT_OFF, NOTE_QUAL_OFF, NOTE_NONE))

    note_tokens = []
    last_pos = -1
    for pos, pitch, dur, vel in sorted(events, key=lambda e: (e[0], e[1])):
        if pos != last_pos:
            note_tokens.append(NOTE_POS_OFF + min(pos, POSITIONS - 1))
            last_pos = pos
        note_tokens.append(NOTE_ON_OFF + (pitch - MIN_PITCH))
        note_tokens.append(NOTE_DUR_OFF + (dur - 1))
        note_tokens.append(NOTE_VEL_OFF + vel)
    note_tokens.append(NOTE_BAR_END)
    return prefix, note_tokens


# ──────────────────────────────────────────────
#  Public decoding functions
# ──────────────────────────────────────────────
def decode_chord_sequence(tokens):
    """Token list → list of (root, qual) or None, one entry per bar."""
    chords = []
    i = 1 if tokens and tokens[0] == CHORD_BOS else 0
    while i + 1 < len(tokens):
        t1, t2 = tokens[i], tokens[i + 1]
        if t1 == CHORD_EOS or t2 == CHORD_EOS:
            break
        if t1 == CHORD_NONE or not (CHORD_ROOT_OFF <= t1 < CHORD_QUAL_OFF):
            chords.append(None)
        elif CHORD_QUAL_OFF <= t2 < CHORD_NONE:
            chords.append((t1 - CHORD_ROOT_OFF, t2 - CHORD_QUAL_OFF))
        else:
            chords.append(None)
        i += 2
    return chords


def decode_bar_notes(note_tokens):
    """Note token list → list of (pos, pitch, dur, vel_bin)."""
    events, cur_pos = [], 0
    i = 0
    while i < len(note_tokens):
        tok = note_tokens[i]
        if tok in (NOTE_BAR_END, NOTE_PAD):
            break
        if NOTE_POS_OFF <= tok < NOTE_ON_OFF:
            cur_pos = tok - NOTE_POS_OFF
        elif NOTE_ON_OFF <= tok < NOTE_DUR_OFF and i + 2 < len(note_tokens):
            dt, vt = note_tokens[i + 1], note_tokens[i + 2]
            if NOTE_DUR_OFF <= dt < NOTE_VEL_OFF and NOTE_VEL_OFF <= vt < NOTE_VOCAB:
                pitch = tok - NOTE_ON_OFF + MIN_PITCH
                dur   = dt - NOTE_DUR_OFF + 1
                vel   = vt - NOTE_VEL_OFF
                events.append((cur_pos, pitch, dur, vel))
                i += 2
        i += 1
    return events


# ──────────────────────────────────────────────
#  Assemble bars → MIDI file
# ──────────────────────────────────────────────
def bars_to_midi(bar_events_list, bpm=90, ticks_per_beat=480):
    mid   = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))

    t16 = ticks_per_beat // 4
    raw = []
    for bar_idx, events in enumerate(bar_events_list):
        bar_start = bar_idx * POSITIONS
        for pos, pitch, dur, vel_bin in events:
            t_on  = (bar_start + pos) * t16
            t_off = t_on + dur * t16
            raw.append((t_on,  'on',  pitch, _bin_vel(vel_bin)))
            raw.append((t_off, 'off', pitch, 0))

    # Sort: off before on at same tick (prevents stuck notes)
    raw.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))

    prev = 0
    for tick, kind, pitch, vel in raw:
        delta = tick - prev
        if kind == 'on':
            track.append(mido.Message('note_on',  note=pitch, velocity=vel, time=delta))
        else:
            track.append(mido.Message('note_off', note=pitch, velocity=0,   time=delta))
        prev = tick
    return mid


# ──────────────────────────────────────────────
#  Full pipeline: one MIDI → training data
# ──────────────────────────────────────────────
def midi_to_training_data(path):
    """
    Returns:
        chord_seq   : token list for chord model  (or None on failure)
        bar_samples : list of (prefix, note_tokens) for note model
    """
    notes = _parse_midi(path)
    if not notes:
        return None, None

    bars = _notes_to_bars(notes)
    if not bars:
        return None, None

    n_bars = max(bars) + 1
    if n_bars < 2:
        return None, None

    chords    = {b: _best_chord([p for _, p, _, _ in evs]) for b, evs in bars.items()}
    chord_seq = encode_chord_sequence(chords, n_bars)

    bar_samples      = []
    prev_bar_tokens  = None   # raw note tokens from the previous bar
    for b in range(n_bars):
        evs = bars.get(b)
        if not evs:
            prev_bar_tokens = None   # reset on gap — no misleading context
            continue
        prefix, note_tokens = encode_bar(
            chords.get(b - 1) if b > 0 else None,
            chords.get(b),
            evs,
            prev_bar_tokens=prev_bar_tokens,
        )
        bar_samples.append((prefix, note_tokens))
        prev_bar_tokens = note_tokens   # carry forward for next bar

    return chord_seq, bar_samples
