#!/usr/bin/env python3
"""
MIDI cohesion analysis script.
Diagnoses why output sounds like "really good improv" rather than composed music.
Analyses:
  1. Pitch class usage over time (4 segments)
  2. Interval distribution (melodic leap/step ratio)
  3. Rhythmic/duration pattern consistency
  4. Register stability vs drift (avg pitch per bar)
  5. Phrase repetition detection (4-bar windows, ±2 semitone tolerance)
  6. Harmonic rhythm (chord root change frequency / regularity)
"""

import sys
import math
import collections
from pathlib import Path

try:
    import mido
except ImportError:
    print("mido not found — use the correct venv")
    sys.exit(1)

FILES = [
    "/home/thewindmage/Desktop/ElegyBox/ElegyBox-site/songs/beethoven_20260515_004524.mid",
    "/home/thewindmage/Desktop/ElegyBox/ElegyBox-site/songs/beethoven_20260515_004616.mid",
    "/home/thewindmage/Desktop/ElegyBox/ElegyBox-site/songs/classic_aba_20260515_004217.mid",
]

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def pitch_name(p):
    return f"{NOTE_NAMES[int(p) % 12]}{int(p) // 12 - 1}"

# ---------------------------------------------------------------------------
def build_tempo_map(mid):
    """Return list of (abs_tick, tempo_us) sorted by tick."""
    tempo_map = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'set_tempo':
                tempo_map.append((abs_tick, msg.tempo))
    tempo_map.sort(key=lambda x: x[0])
    if not tempo_map or tempo_map[0][0] > 0:
        tempo_map.insert(0, (0, 500000))
    return tempo_map

def ticks_to_seconds(tick, tempo_map, tpb):
    secs = 0.0
    prev_tick = 0
    prev_tempo = 500000
    for t_tick, t_tempo in tempo_map:
        if t_tick >= tick:
            break
        seg_end = min(t_tick, tick)
        secs += (seg_end - prev_tick) * prev_tempo / 1e6 / tpb
        prev_tick = t_tick
        prev_tempo = t_tempo
    secs += (tick - prev_tick) * prev_tempo / 1e6 / tpb
    return secs

# ---------------------------------------------------------------------------
def collect_notes(mid):
    """
    Returns list of (tick_on, tick_off, pitch, velocity, channel).
    Handles multi-track files correctly by merging all tracks.
    """
    # Merge all tracks into a single event stream with absolute ticks
    all_events = []
    for track_idx, track in enumerate(mid.tracks):
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type in ('note_on', 'note_off'):
                all_events.append((abs_tick, track_idx, msg))
    all_events.sort(key=lambda x: (x[0], x[1]))

    active = {}   # (channel, pitch) -> (tick_on, velocity)
    notes = []

    for abs_tick, track_idx, msg in all_events:
        if msg.type == 'note_on' and msg.velocity > 0:
            key = (msg.channel, msg.note)
            active[key] = (abs_tick, msg.velocity)
        else:  # note_off or note_on vel=0
            key = (msg.channel, msg.note)
            if key in active:
                tick_on, vel = active.pop(key)
                notes.append((tick_on, abs_tick, msg.note, vel, msg.channel))

    # Close any still-active notes
    if all_events:
        last_tick = max(e[0] for e in all_events)
        for (ch, pitch), (tick_on, vel) in active.items():
            notes.append((tick_on, last_tick, pitch, vel, ch))

    notes.sort(key=lambda n: n[0])
    return notes

# ---------------------------------------------------------------------------
def analyze_file(filepath):
    path = Path(filepath)
    if not path.exists():
        print(f"\n  FILE NOT FOUND: {filepath}")
        return

    mid = mido.MidiFile(filepath)
    tpb = mid.ticks_per_beat
    tempo_map = build_tempo_map(mid)
    notes = collect_notes(mid)

    print(f"\n{'='*70}")
    print(f"FILE: {path.name}")
    print(f"  Type={mid.type}  Tracks={len(mid.tracks)}  TPB={tpb}  "
          f"BPM={60_000_000/tempo_map[0][1]:.1f}")
    print(f"{'='*70}")

    if not notes:
        print("  No notes found!")
        return

    total_ticks = max(n[1] for n in notes)
    total_seconds = ticks_to_seconds(total_ticks, tempo_map, tpb)
    pitch_min = min(n[2] for n in notes)
    pitch_max = max(n[2] for n in notes)
    print(f"  Duration: {total_seconds:.1f}s  |  Total notes: {len(notes)}")
    print(f"  Pitch range: {pitch_name(pitch_min)} ({pitch_min}) - "
          f"{pitch_name(pitch_max)} ({pitch_max})")

    bar_ticks = tpb * 4   # 4/4 time assumed

    # =========================================================
    # 1. PITCH CLASS USAGE OVER TIME (4 segments)
    # =========================================================
    print(f"\n{'─'*60}")
    print("1. PITCH CLASS USAGE OVER TIME (4 segments)")
    print(f"{'─'*60}")

    seg_counts = [collections.Counter() for _ in range(4)]
    for tick_on, tick_off, pitch, vel, ch in notes:
        seg = min(3, int(tick_on * 4 / total_ticks))
        seg_counts[seg][pitch % 12] += 1

    dominant_pcs = []
    for i, counter in enumerate(seg_counts):
        total_in_seg = sum(counter.values())
        top3 = counter.most_common(3)
        dominant_pcs.append(top3[0][0] if top3 else None)
        top3_str = ", ".join(f"{NOTE_NAMES[pc]}({cnt})" for pc, cnt in top3)
        print(f"  Seg {i+1} [{total_in_seg} notes]: {top3_str}")

    print()
    if all(pc is not None for pc in dominant_pcs):
        unique_dominant = set(dominant_pcs)
        print(f"  Dominant PCs across segs: {[NOTE_NAMES[p] for p in dominant_pcs]}")
        if len(unique_dominant) == 1:
            print(f"  -> CONSISTENT tonic: {NOTE_NAMES[dominant_pcs[0]]} throughout")
        elif len(unique_dominant) == 2:
            print(f"  -> Mostly stable tonic (2 different dominant PCs) — minor drift")
        else:
            print(f"  -> DRIFTING tonic: {len(unique_dominant)} different dominant PCs  [IMPROV INDICATOR]")

    early_pcs = set(pc for pc, _ in seg_counts[0].most_common(4))
    late_pcs  = set(pc for pc, _ in seg_counts[3].most_common(4))
    returning = early_pcs & late_pcs
    print(f"\n  Top-4 PCs seg 1: {[NOTE_NAMES[p] for p in sorted(early_pcs)]}")
    print(f"  Top-4 PCs seg 4: {[NOTE_NAMES[p] for p in sorted(late_pcs)]}")
    if returning:
        print(f"  -> Returning PCs (motivic return): {[NOTE_NAMES[p] for p in sorted(returning)]}")
    else:
        print(f"  -> NO pitch class return from seg1→seg4  [IMPROV INDICATOR]")

    # =========================================================
    # 2. INTERVAL DISTRIBUTION
    # =========================================================
    print(f"\n{'─'*60}")
    print("2. INTERVAL DISTRIBUTION (melodic intervals in highest voice)")
    print(f"{'─'*60}")

    # Track highest concurrent pitch at each note_on event
    cur_active = collections.Counter()
    prev_melody = None
    intervals = []

    # Rebuild event stream in order
    all_events = []
    for track_idx, track in enumerate(mid.tracks):
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type in ('note_on', 'note_off'):
                all_events.append((abs_tick, track_idx, msg))
    all_events.sort(key=lambda x: (x[0], 0 if (x[2].type == 'note_off' or
                                                  (x[2].type == 'note_on' and x[2].velocity == 0)) else 1))

    for abs_tick, track_idx, msg in all_events:
        if msg.type == 'note_on' and msg.velocity > 0:
            cur_active[msg.note] += 1
            top = max(cur_active.keys())
            if prev_melody is not None:
                intervals.append(top - prev_melody)
            prev_melody = top
        else:
            cur_active[msg.note] = max(0, cur_active[msg.note] - 1)
            if cur_active[msg.note] == 0:
                del cur_active[msg.note]

    if not intervals:
        print("  No intervals computed.")
    else:
        ic = collections.Counter(intervals)
        total_int = len(intervals)
        unison    = ic.get(0, 0)
        steps     = sum(ic[i] for i in [-2,-1,1,2])
        sm_leaps  = sum(ic[i] for i in [-4,-3,3,4])
        lg_leaps  = sum(v for i,v in ic.items() if abs(i) >= 5)

        print(f"  Total melodic intervals: {total_int}")
        print(f"  Unison (0):       {unison:4d}  ({100*unison/total_int:5.1f}%)")
        print(f"  Steps  (±1-2):    {steps:4d}  ({100*steps/total_int:5.1f}%)")
        print(f"  Small leaps(±3-4):{sm_leaps:4d}  ({100*sm_leaps/total_int:5.1f}%)")
        print(f"  Large leaps(≥±5): {lg_leaps:4d}  ({100*lg_leaps/total_int:5.1f}%)")

        top8 = ic.most_common(8)
        top8_str = "  ".join(f"{i:+d}({c})" for i,c in sorted(top8, key=lambda x: x[0]))
        print(f"  Top intervals: {top8_str}")

        lg_pct = lg_leaps / total_int
        sw_pct = (steps + unison) / total_int
        if lg_pct > 0.35:
            print(f"  -> HIGH leap ratio ({100*lg_pct:.1f}%)  [IMPROV INDICATOR — random jumping]")
        elif lg_pct > 0.20:
            print(f"  -> Elevated leap ratio ({100*lg_pct:.1f}%) — leapier than typical composed melody")
        elif sw_pct > 0.65:
            print(f"  -> Mostly stepwise ({100*sw_pct:.1f}%) — good melodic contour")
        else:
            print(f"  -> Mixed motion (leaps={100*lg_pct:.1f}%, stepwise={100*sw_pct:.1f}%)")

    # =========================================================
    # 3. RHYTHMIC PATTERN CONSISTENCY
    # =========================================================
    print(f"\n{'─'*60}")
    print("3. RHYTHMIC PATTERN CONSISTENCY")
    print(f"{'─'*60}")

    durations_raw = [n[1] - n[0] for n in notes if n[1] > n[0]]
    if durations_raw:
        sixteenth = max(1, tpb // 4)
        quantized = [max(1, round(d / sixteenth)) for d in durations_raw]
        dc = collections.Counter(quantized)

        mn = min(quantized); mx = max(quantized)
        mean_dur = sum(quantized) / len(quantized)
        print(f"  Durations (1/16th note units): min={mn}  max={mx}  mean={mean_dur:.2f}  unique={len(dc)}")

        top6 = dc.most_common(6)
        top_str = "  ".join(f"{d}x16th({c})" for d,c in top6)
        print(f"  Top durations: {top_str}")

        top3_sum = sum(c for _,c in top6[:3])
        concentration = top3_sum / len(quantized)
        print(f"  Top-3 concentration: {100*concentration:.1f}% of all notes")

        if concentration > 0.75:
            print(f"  -> HIGH concentration — consistent rhythmic pulse  [composed feel]")
        elif concentration > 0.55:
            print(f"  -> Moderate rhythmic variety")
        else:
            print(f"  -> LOW concentration ({100*concentration:.1f}%)  [IMPROV INDICATOR — random durations]")

        # Repeating 4-note rhythmic cells
        cell_len = 4
        if len(quantized) >= cell_len * 2:
            cells = [tuple(quantized[i:i+cell_len]) for i in range(len(quantized)-cell_len+1)]
            cell_counter = collections.Counter(cells)
            repeated = [(c,n) for c,n in cell_counter.most_common(5) if n > 1]
            if repeated:
                print(f"  Repeating 4-note rhythmic cells:")
                for cell, count in repeated[:3]:
                    print(f"    {cell} appears {count} times")
            else:
                print(f"  -> NO repeating 4-note rhythmic cells  [IMPROV INDICATOR]")

    # =========================================================
    # 4. REGISTER STABILITY VS DRIFT
    # =========================================================
    print(f"\n{'─'*60}")
    print("4. REGISTER STABILITY (avg pitch per bar)")
    print(f"{'─'*60}")

    num_bars = max(1, int(total_ticks / bar_ticks) + 1)
    bar_pitches = [[] for _ in range(num_bars)]
    for tick_on, tick_off, pitch, vel, ch in notes:
        bar = int(tick_on / bar_ticks)
        if bar < num_bars:
            bar_pitches[bar].append(pitch)

    bar_avgs = [sum(bp)/len(bp) for bp in bar_pitches if bp]

    if len(bar_avgs) >= 2:
        global_avg = sum(bar_avgs) / len(bar_avgs)
        diffs = [abs(bar_avgs[i+1] - bar_avgs[i]) for i in range(len(bar_avgs)-1)]
        avg_diff = sum(diffs) / len(diffs)
        max_diff = max(diffs)
        total_range = max(bar_avgs) - min(bar_avgs)

        print(f"  Bars analyzed: {len(bar_avgs)}")
        print(f"  Global avg pitch: {global_avg:.1f} ({pitch_name(round(global_avg))})")
        print(f"  Avg bar-to-bar change: {avg_diff:.1f} semitones")
        print(f"  Max bar-to-bar jump:   {max_diff:.1f} semitones")
        print(f"  Overall range across bars: {total_range:.1f} semitones")

        # Show bar-by-bar plot (first 24 bars)
        disp = bar_avgs[:24]
        bar_str = " ".join(f"{a:.0f}" for a in disp)
        print(f"  First {len(disp)} bar avg pitches: [{bar_str}]")

        if avg_diff > 8:
            print(f"  -> WILDLY FLUCTUATING register  [IMPROV INDICATOR]")
        elif avg_diff > 4:
            print(f"  -> Moderate register instability ({avg_diff:.1f} st avg change)")
        else:
            print(f"  -> Stable register ({avg_diff:.1f} st avg change)  [composed feel]")

    # =========================================================
    # 5. PHRASE REPETITION DETECTION (4-bar windows)
    # =========================================================
    print(f"\n{'─'*60}")
    print("5. PHRASE REPETITION DETECTION (4-bar windows, ±2 st, ±0.3 beat)")
    print(f"{'─'*60}")

    bar_ticks_f = float(bar_ticks)
    window_bars = 4

    def notes_in_bar_range(start_bar, n_bars):
        end_bar = start_bar + n_bars
        result = []
        for tick_on, tick_off, pitch, vel, ch in notes:
            b = tick_on / bar_ticks_f
            if start_bar <= b < end_bar:
                result.append((b - start_bar, pitch))
        return result

    def phrase_similarity(phrase_a, phrase_b, pitch_tol=2, time_tol=0.3):
        if not phrase_a or not phrase_b:
            return 0.0
        matches = 0
        for t_a, p_a in phrase_a:
            for t_b, p_b in phrase_b:
                if abs(t_a - t_b) < time_tol and abs(p_a - p_b) <= pitch_tol:
                    matches += 1
                    break
        return matches / max(len(phrase_a), len(phrase_b))

    num_4bar_windows = max(1, num_bars // window_bars)
    phrases = [(w * window_bars, notes_in_bar_range(w * window_bars, window_bars))
               for w in range(num_4bar_windows)]

    repetitions = []
    for i in range(len(phrases)):
        for j in range(i+2, len(phrases)):   # skip adjacent
            sa, pa = phrases[i]
            sb, pb = phrases[j]
            if len(pa) < 4 or len(pb) < 4:
                continue
            sim = phrase_similarity(pa, pb)
            if sim >= 0.55:
                repetitions.append((sa, sb, sim, len(pa), len(pb)))

    print(f"  4-bar windows checked: {len(phrases)} windows, "
          f"{len(phrases)*(len(phrases)-1)//2} pairs")
    if repetitions:
        repetitions.sort(key=lambda x: -x[2])
        print(f"  Repetitions found (sim ≥ 55%):")
        for sa, sb, sim, la, lb in repetitions[:6]:
            print(f"    Bars {sa+1:2d}-{sa+window_bars} ≈ Bars {sb+1:2d}-{sb+window_bars}  "
                  f"sim={100*sim:.0f}%  ({la} vs {lb} notes)")
        print(f"  -> {len(repetitions)} phrase repetition(s) — structured feel")
    else:
        print(f"  -> NO phrase repetitions found  [IMPROV INDICATOR — no thematic return]")

    # =========================================================
    # 6. HARMONIC RHYTHM
    # =========================================================
    print(f"\n{'─'*60}")
    print("6. HARMONIC RHYTHM (root change rate and regularity)")
    print(f"{'─'*60}")

    half_bar = tpb * 2
    num_hb = max(1, int(total_ticks / half_bar) + 1)
    hb_pc = [collections.Counter() for _ in range(num_hb)]

    for tick_on, tick_off, pitch, vel, ch in notes:
        hb = int(tick_on / half_bar)
        if hb < num_hb:
            hb_pc[hb][pitch % 12] += 1

    hb_roots = [c.most_common(1)[0][0] if c else None for c in hb_pc]
    hb_roots = [r for r in hb_roots if r is not None]

    if len(hb_roots) >= 3:
        changes = sum(1 for i in range(1, len(hb_roots)) if hb_roots[i] != hb_roots[i-1])
        change_rate = changes / len(hb_roots)
        print(f"  Half-bar windows: {len(hb_roots)}")
        print(f"  Root changes: {changes}  ({100*change_rate:.1f}% of windows change)")

        change_pos = [i for i in range(1, len(hb_roots)) if hb_roots[i] != hb_roots[i-1]]
        if len(change_pos) > 1:
            gaps = [change_pos[i+1] - change_pos[i] for i in range(len(change_pos)-1)]
            gap_mean = sum(gaps) / len(gaps)
            gap_var  = sum((g-gap_mean)**2 for g in gaps) / len(gaps)
            gap_std  = math.sqrt(gap_var)
            cv = gap_std / max(gap_mean, 0.01)
            print(f"  Gap between changes: mean={gap_mean:.1f} half-bars  std={gap_std:.1f}  CV={cv:.2f}")
            if cv < 0.4:
                print(f"  -> REGULAR harmonic rhythm (CV={cv:.2f})  [composed feel]")
            elif cv < 0.8:
                print(f"  -> Somewhat irregular harmonic rhythm (CV={cv:.2f})")
            else:
                print(f"  -> VERY IRREGULAR harmonic rhythm (CV={cv:.2f})  [IMPROV INDICATOR]")

        # Show root progression for first 24 half-bars
        roots_str = " ".join(NOTE_NAMES[r] for r in hb_roots[:24])
        print(f"  First 24 half-bar roots: {roots_str}")

        # Count how many distinct roots appear
        distinct_roots = len(set(hb_roots))
        print(f"  Distinct roots used: {distinct_roots}/12 pitch classes")
        if distinct_roots > 7:
            print(f"  -> Many different chord roots ({distinct_roots}) — chromatically wandering  [IMPROV INDICATOR]")
        elif distinct_roots <= 4:
            print(f"  -> Few distinct roots ({distinct_roots}) — harmonically focused  [composed feel]")

    # =========================================================
    # SUMMARY BOX
    # =========================================================
    print(f"\n{'─'*60}")
    print("IMPROV INDICATORS SUMMARY")
    print(f"{'─'*60}")


# ---------------------------------------------------------------------------
def main():
    for f in FILES:
        analyze_file(f)

    print(f"\n\n{'='*70}")
    print("GLOBAL DIAGNOSIS: IMPROV vs COMPOSED")
    print(f"{'='*70}")
    print("""
WHAT MAKES MUSIC SOUND COMPOSED (not improvised):
  - Consistent tonic or purposeful modulation arc
  - Stepwise melody; leaps are motivically justified
  - Repeating rhythmic cells / ostinato patterns
  - Register moves in arcs (not random bar-to-bar jumps)
  - Phrase repetition: ABA, AABB, development-recapitulation
  - Regular, predictable harmonic rhythm (every 2 or 4 bars)
  - Few distinct harmonic roots (diatonic focus, not chromatic drift)

KEY IMPROV INDICATORS TO LOOK FOR:
  - Drifting dominant PC across segments
  - Large-leap ratio > 25-35%
  - No repeating 4-note rhythmic cells
  - Avg bar-to-bar pitch change > 5-6 semitones
  - No phrase repetitions at 55%+ similarity
  - Harmonic rhythm CV > 0.8 (chaotic chord change timing)
  - More than 7 distinct chord roots used
""")

if __name__ == '__main__':
    main()
