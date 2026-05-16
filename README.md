# ElegyBox

AI-powered classical piano music generation. A two-stage GPT pipeline trained on the [Mutopia Project](https://www.mutopiaproject.org/) corpus generates original MIDI compositions in the style of Bach, Chopin, Mozart, Beethoven, and others. A lightweight Flask web app serves inference with a piano-roll visualizer and in-browser playback.

---

## Quick Start — Run the Web App

Assumes you have pre-trained checkpoints (`chord_model.pt`, `note_model_hierarchical.pt`, `bar_encoder.pt`) in `checkpoints/`.

```bash
git clone https://github.com/your-username/ElegyBox.git
cd ElegyBox

# Install dependencies (Python 3.10+)
pip install torch mido tqdm numpy flask

# Start the server
python ElegyBox-site/app.py
```

Open **http://localhost:5173** in your browser. Pick a preset, click **Generate**, and the piece streams in. Finished songs appear in the library for playback, download, rename, or deletion.

> **GPU note:** `generate.py` auto-detects CUDA. CPU inference works but takes ~30 s per piece. With a GPU it's ~3–5 s.

---

## Project Structure

```
ElegyBox/
├── checkpoints/               # Trained model weights (not in repo — see below)
│   ├── chord_model.pt
│   ├── note_model.pt          # base note model (Stage 1)
│   ├── note_model_hierarchical.pt  # fine-tuned with BarEncoder (Stage 2)
│   └── bar_encoder.pt
├── data/
│   ├── mutopia/               # raw MIDI corpus (created by download_mutopia.py)
│   └── processed/             # tokenised pkl files (created by preprocess*.py)
├── ElegyBox-site/
│   ├── app.py                 # Flask server
│   ├── static/                # CSS + JS frontend
│   ├── templates/index.html
│   └── songs/                 # generated MIDIs stored here
├── tokenizer.py               # shared tokenisation & MIDI I/O
├── models.py                  # MusicGPT (chord + note transformer)
├── models_hierarchical.py     # HierarchicalMusicGPT + BarEncoder
├── dataset.py                 # PyTorch datasets
├── preprocess.py              # tokenise corpus → base pkl files
├── preprocess_hierarchical.py # build hierarchical training samples
├── train.py                   # train chord model + base note model
├── train_hierarchical.py      # fine-tune with BarEncoder (Phase 1 + 2)
├── generate.py                # inference script (CLI)
└── download_mutopia.py        # scrape & download Mutopia MIDI corpus
```

---

## Full Training Pipeline

### 1. Download the corpus

```bash
pip install requests beautifulsoup4
python download_mutopia.py
```

This scrapes the Mutopia FTP index and downloads all MIDI zips into `data/mutopia/<Composer>/<Piece>/`. Expect ~2,000–3,000 files and ~30–60 minutes depending on your connection.

---

### 2. Preprocess — base tokenisation

```bash
python preprocess.py
```

Reads every `.mid`/`.midi` file under `data/`, tokenises it with the shared vocabulary, and saves two files:

| File | Contents |
|---|---|
| `data/processed/chord_sequences.pkl` | Token lists for the chord model |
| `data/processed/note_samples.pkl` | `(prefix, note_tokens)` pairs for the base note model |

Typical output: ~1,800 valid files, ~250,000 note samples.

---

### 3. Train the chord model and base note model

```bash
python train.py                          # trains both models (default)
python train.py --stage chords           # chord model only
python train.py --stage notes            # note model only
```

Key arguments:

| Arg | Default | Description |
|---|---|---|
| `--chord_epochs` | 120 | Epochs for the chord model |
| `--note_epochs` | 80 | Epochs for the base note model |
| `--batch_size` | 64 | Batch size (reduce if OOM) |
| `--lr` | 3e-4 | Peak learning rate (cosine schedule with warmup) |
| `--val_split` | 0.05 | Fraction held out for validation |

Checkpoints save to `checkpoints/chord_model.pt` and `checkpoints/note_model.pt` whenever validation loss improves.

**Model sizes:**

| Model | d_model | Heads | Layers | seq_len | Params |
|---|---|---|---|---|---|
| Chord GPT | 128 | 4 | 4 | 256 | ~1 M |
| Note GPT (base) | 384 | 8 | 8 | 320 | ~15 M |

Training on a single T4 GPU: chord model ~1 h, note model ~4–6 h.

---

### 4. Preprocess — hierarchical training data

Run locally before uploading to a cloud GPU:

```bash
python preprocess_hierarchical.py
```

Builds `data/processed/note_samples_hierarchical.pkl`: each sample now includes up to 16 bars of encoded history alongside the current bar's prefix and note tokens. Key transposition augmentation (±6 semitones) is applied during training, not here.

Options:

| Arg | Default | Description |
|---|---|---|
| `--data_dir` | `data/` | Root to search for MIDI files |
| `--max_history` | 16 | Max history bars per sample |

---

### 5. Fine-tune — HierarchicalMusicGPT + BarEncoder

The hierarchical training adds long-range memory via cross-attention. A small **BarEncoder** (bidirectional transformer) compresses each completed bar into 4 summary vectors (384-dim) and feeds them into every layer of the note model as cross-attention context.

Architecture additions:

| Component | Size | Role |
|---|---|---|
| BarEncoder | d=128, 4 heads, 3 layers | Encodes one bar → 4×384 summary vectors |
| Cross-attention adapters | per note model layer | Attends to BarEncoder summaries |
| Max history | 16 bars | 64 summary vectors in memory at once |

Training is split into two phases, designed to be run on a cloud GPU (e.g. Lightning AI T4):

```bash
python train_hierarchical.py
```

**Phase 1** (~15 epochs, LR 3e-4): the existing note model is frozen. Only the BarEncoder and the new cross-attention adapters are trained.

**Phase 2** (~20 epochs): everything is unfrozen and fine-tuned jointly with a split learning rate — 5e-6 for the original note model weights (very conservative, preserves learned distribution), 1e-4 for the adapters and BarEncoder.

Checkpoints: `checkpoints/note_model_hierarchical.pt` and `checkpoints/bar_encoder.pt`.

> The hierarchical model is a strict drop-in for the base note model. If `note_model_hierarchical.pt` is absent, inference falls back to `note_model.pt` automatically (`memory=None` disables cross-attention).

---

## CLI Inference

`generate.py` is the full inference script. The web app calls it as a subprocess.

```bash
# Minimal — 32 bars, 90 BPM, ABA form
python generate.py --output piece.mid

# Chopin nocturne style
python generate.py --output nocturne.mid --n_bars 40 --bpm 72 --form aba \
  --note_temp 0.95 --chord_temp 0.9 --max_chord_notes 5 --max_repeat 3 \
  --motif_strength 0.65 --melody_strength 0.4 --dur_bias 0.4

# Use a hand-crafted chord file (bypasses chord model)
python generate.py --chord_file ragtime_chords.json --n_bars 32 --bpm 108

# Seed the bar encoder with context from an existing MIDI
python generate.py --seed_midi data/mutopia/VivaldiA/spring/spring-score.mid \
  --seed_bars 8 --form arch --output vivaldi_style.mid
```

### Key parameters

**Structure**

| Arg | Default | Description |
|---|---|---|
| `--n_bars` | 32 | Total bars to generate |
| `--bpm` | 90 | Tempo |
| `--form` | `none` | Musical form: `none` / `aba` / `binary` / `rondo` / `arch` / `variation` |
| `--phrase_bars` | 4 | Phrase length in bars |

**Sampling**

| Arg | Default | Description |
|---|---|---|
| `--note_temp` | 1.0 | Note model temperature |
| `--chord_temp` | 1.1 | Chord model temperature |
| `--top_k` | 40 | Top-k nucleus (note model) |
| `--top_p` | 0.9 | Top-p nucleus (note model) |
| `--b_note_temp` | 0.0 | Temperature for B/C sections (0 = same as A) |

**Musical biases**

| Arg | Default | Description |
|---|---|---|
| `--key_strength` | 0.0 | Diatonic pull on note pitches |
| `--chord_tone_strength` | 0.0 | Extra pull toward current chord tones |
| `--motif_strength` | 0.0 | Pitch-class coherence from opening bars |
| `--melody_strength` | 0.0 | Stepwise continuity in the top voice |
| `--pitch_repeat_penalty` | 1.5 | Penalty for re-attacking the same pitch consecutively |
| `--dur_repeat_penalty` | 0.8 | Penalty for repeating the same duration |
| `--vel_arc` | 0.0 | Global dynamic arc (quiet→loud→quiet) |

**Texture**

| Arg | Default | Description |
|---|---|---|
| `--max_chord_notes` | 5 | Max simultaneous notes per bar |
| `--max_repeat` | 3 | Max consecutive repeated chord roots |
| `--lh_pattern` | `none` | Left-hand pattern: `alberti` / `stride` / `waltz` / `walking` / `murky` |
| `--lh_bass_split` | 58 | MIDI pitch below which LH pattern applies |

**Humanization** (on by default)

| Arg | Default | Description |
|---|---|---|
| `--jitter_ms` | 11.0 | Timing jitter std-dev in ms (chords shift as a unit) |
| `--no_humanize` | — | Disable timing jitter, phrase arcs, and melodic contour shaping |

**Seeding**

| Arg | Default | Description |
|---|---|---|
| `--chord_file` | — | JSON `[[root, qual], …]` list to bypass the chord model |
| `--seed_midi` | — | MIDI file whose bars pre-populate the BarEncoder history |
| `--seed_bars` | 8 | How many bars to take from the seed MIDI (max 16) |

**Flags**

| Arg | Description |
|---|---|
| `--raw` | Skip all post-processing; output raw model tokens |
| `--no_bias` | Zero all musical biases (pure model distribution) |
| `--debug_stages` | Save intermediate MIDIs at each post-processing step |

---

## Vocabulary

The shared tokeniser uses two separate vocabularies.

**Chord vocabulary** (26 tokens):

| Range | Meaning |
|---|---|
| 0 | PAD |
| 1 | BOS |
| 2 | EOS |
| 3–14 | Root (C=3 … B=14) |
| 15–24 | Quality (maj, min, dim, aug, dom7, maj7, min7, hdim7, sus2, sus4) |
| 25 | No chord |

**Note vocabulary** (153 tokens):

| Range | Meaning |
|---|---|
| 0 | PAD |
| 1 | BAR\_END |
| 2–5 | Chord context (root × 2, quality × 2) |
| 6 | No chord |
| 7–38 | Position (16th-note slot 0–31) |
| 39–126 | Note pitch (MIDI 21–108) |
| 127–143 | Duration (1–17 sixteenth-note slots) |
| 145–152 | Velocity (8 bins, ~8–104 MIDI) |

Each note is encoded as four consecutive tokens: **position → pitch → duration → velocity**.

---

## Chord File Format

To bypass the chord model, pass `--chord_file` pointing to a JSON array of `[root, quality]` pairs — one entry per bar:

```json
[
  [0, 0], [0, 0], [7, 4], [7, 4],
  [0, 4], [5, 0], [2, 1], [0, 0]
]
```

Root encoding: C=0, C#=1, D=2, D#=3, E=4, F=5, F#=6, G=7, G#=8, A=9, A#=10, B=11.

Quality encoding: maj=0, min=1, dim=2, aug=3, dom7=4, maj7=5, min7=6, hdim7=7, sus2=8, sus4=9.

If the file has fewer entries than `--n_bars`, the last chord is repeated to fill.

---

## Humanization

By default every generated MIDI goes through three post-processing passes:

1. **Phrase arcs** — each 4-bar phrase gets a smooth velocity envelope (soft start, swell into bar 3, taper at bar 4), mimicking the natural breath of a musical sentence.
2. **Melodic contour** — treble notes at higher pitch within a bar receive up to +6 velocity bins; lower notes receive −3 (natural singing tendency). Longer notes receive an additional +1 agogic accent.
3. **Timing jitter** — each rhythmic position is shifted by a Gaussian offset (default σ = 11 ms). All notes at the same position shift together so chords land as a unit. Strong beats get 40% of the jitter of weak beats. The final 25% of each phrase receives a slight rallentando push.

Pass `--no_humanize` to get a metronomic MIDI.

---

## Hardware Requirements

| Task | Minimum | Recommended |
|---|---|---|
| Inference (web app) | CPU | Any CUDA GPU |
| Base training | 8 GB VRAM | 16 GB VRAM (T4 / A10) |
| Hierarchical fine-tuning | 10 GB VRAM | 16 GB VRAM |

Mixed precision (`torch.amp`) is enabled automatically when CUDA is available.