'use strict';

// ── Baseline generation params (matched to what sounds good at current training stage) ──
// Presets override individual keys; anything not in a preset uses these defaults.
const BASE_PARAMS = {
  top_k: 30,
  top_p: 0.85,
  motif_strength: 0.65,
  key_strength: 0.2,
  coherence_decay: 0.1,
};

// ── Presets ────────────────────────────────────────────────────────────────
const PRESETS = [

  // ── Base ────────────────────────────────────────────────────────────────
  { name: 'Clean',          category: 'Base',
    description: 'Balanced and uncoloured — a reliable starting point.',
    params: { n_bars:32, bpm:96,  chord_temp:1.0,  note_temp:1.0,  max_chord_notes:5, max_repeat:3 } },
  { name: 'Slow & Lyrical', category: 'Base',
    description: 'Gentle tempo, long singing lines, sustained legato.',
    params: { n_bars:32, bpm:64,  chord_temp:0.9,  note_temp:0.9,  max_chord_notes:4, max_repeat:3,
              dur_bias:0.6, melody_strength:0.5 } },
  { name: 'Bright & Lively',category: 'Base',
    description: 'Fast tempo, lighter texture, crisp articulation.',
    params: { n_bars:32, bpm:116, chord_temp:0.95, note_temp:1.0,  max_chord_notes:4, max_repeat:2 } },
  { name: 'Dense & Rich',   category: 'Base',
    description: 'Full voicings, warm romantic texture, unhurried.',
    params: { n_bars:32, bpm:88,  chord_temp:0.9,  note_temp:0.95, max_chord_notes:5, max_repeat:3,
              dur_bias:0.3, melody_strength:0.3 } },
  { name: 'Sparse',         category: 'Base',
    description: 'Two or three voices — open, transparent, intimate.',
    params: { n_bars:32, bpm:80,  chord_temp:0.9,  note_temp:0.9,  max_chord_notes:3, max_repeat:3,
              dur_bias:0.4, melody_strength:0.4 } },

  // ── Structured ──────────────────────────────────────────────────────────
  { name: 'Binary (AB)',    category: 'Structured',
    description: 'Two contrasting halves — grounded A section, freer B section.',
    params: { n_bars:32, bpm:100, chord_temp:0.95, note_temp:0.95, form:'binary', max_chord_notes:5, max_repeat:3, b_note_temp:1.1,
              dur_bias:0.2, motif_strength:0.5 } },
  { name: 'ABA',            category: 'Structured',
    description: 'Theme — contrast — return. The backbone of classical form.',
    params: { n_bars:48, bpm:96,  chord_temp:0.9,  note_temp:0.95, form:'aba',    max_chord_notes:5, max_repeat:3, b_note_temp:1.1,
              dur_bias:0.2, melody_strength:0.3, motif_strength:0.5 } },
  { name: 'Rondo',          category: 'Structured',
    description: 'ABACABA — a recurring refrain between two contrasting episodes.',
    params: { n_bars:56, bpm:104, chord_temp:1.0,  note_temp:0.95, form:'rondo',  max_chord_notes:5, max_repeat:2, b_note_temp:1.1,
              motif_strength:0.5 } },
  { name: 'Arch (ABCBA)',   category: 'Structured',
    description: 'Builds through B to a climactic C, then mirrors back to the opening.',
    params: { n_bars:50, bpm:96,  chord_temp:0.9,  note_temp:0.9,  form:'arch',   max_chord_notes:5, max_repeat:3, b_note_temp:1.05,
              dur_bias:0.2, melody_strength:0.3, motif_strength:0.5 } },
  { name: 'Variations',     category: 'Structured',
    description: 'Three passes over the same chords — each variation denser and freer.',
    params: { n_bars:48, bpm:96,  chord_temp:0.9,  note_temp:0.9,  form:'variation', max_chord_notes:5, max_repeat:3,
              dur_bias:0.2, motif_strength:0.5 } },

  // ── Character ───────────────────────────────────────────────────────────
  { name: 'Nocturne',       category: 'Character',
    description: 'Night music — slow ABA, intimate melody over warm sustained harmonies.',
    params: { n_bars:32, bpm:58,  chord_temp:0.88, note_temp:0.9,  form:'aba',    max_chord_notes:4, max_repeat:3, b_note_temp:1.0,
              dur_bias:0.7, melody_strength:0.5 } },
  { name: 'Elegy',          category: 'Character',
    description: 'Mournful and unhurried — lamenting lines that linger in silence.',
    params: { n_bars:32, bpm:60,  chord_temp:0.88, note_temp:0.88, max_chord_notes:4, max_repeat:3,
              dur_bias:0.6, melody_strength:0.5 } },
  { name: 'Hymn',           category: 'Character',
    description: 'Slow, solemn, chordal — every note placed with gravity and intention.',
    params: { n_bars:24, bpm:58,  chord_temp:0.85, note_temp:0.88, max_chord_notes:5, max_repeat:4,
              dur_bias:0.8 } },
  { name: 'Prelude',        category: 'Character',
    description: 'Flowing, searching — harmonically alive, always pressing forward.',
    params: { n_bars:24, bpm:96,  chord_temp:0.95, note_temp:0.95, max_chord_notes:5, max_repeat:3,
              dur_bias:0.2 } },
  { name: 'Étude',          category: 'Character',
    description: 'Fast binary — one idea pursued relentlessly, A driven, B contrasting.',
    params: { n_bars:32, bpm:120, chord_temp:0.95, note_temp:1.0,  form:'binary', max_chord_notes:4, max_repeat:2, b_note_temp:1.05 } },
  { name: 'Dance',          category: 'Character',
    description: 'Quick, bright, rhythmically alive — light on its feet.',
    params: { n_bars:32, bpm:120, chord_temp:0.95, note_temp:1.0,  max_chord_notes:4, max_repeat:2,
              bass_beat_strength:0.3 } },
  { name: 'March',          category: 'Character',
    description: 'Steady, purposeful binary — strong downbeats, clear phrases.',
    params: { n_bars:32, bpm:108, chord_temp:0.95, note_temp:1.0,  form:'binary', max_chord_notes:4, max_repeat:3, b_note_temp:1.0,
              bass_beat_strength:0.6 } },
  { name: 'Scherzo',        category: 'Character',
    description: 'Fast, playful rondo — impish, light-footed, full of rhythmic wit.',
    params: { n_bars:42, bpm:148, chord_temp:1.0,  note_temp:1.05, form:'rondo',  max_chord_notes:4, max_repeat:2, b_note_temp:1.05,
              dur_bias:-0.2 } },
  { name: 'Toccata',        category: 'Character',
    description: 'Relentless motor-rhythm arch — fast, dense, unyielding start to finish.',
    params: { n_bars:32, bpm:144, chord_temp:1.0,  note_temp:1.05, form:'arch',   max_chord_notes:4, max_repeat:2, b_note_temp:1.05,
              dur_bias:-0.2 } },
  { name: 'Fantasia',       category: 'Character',
    description: 'Free and improvisatory arch — wide range, wandering harmony, expressive sweep.',
    params: { n_bars:40, bpm:88,  chord_temp:1.05, note_temp:1.0,  form:'arch',   max_chord_notes:5, max_repeat:3, b_note_temp:1.05,
              dur_bias:0.3, melody_strength:0.3 } },

  // ── Composer ────────────────────────────────────────────────────────────
  { name: 'Bach Invention', category: 'Composer',
    description: 'Two-voice binary counterpoint — brisk, imitative, each voice independent.',
    params: { n_bars:24, bpm:104, chord_temp:1.0,  note_temp:1.0,  form:'binary', max_chord_notes:2, max_repeat:2, b_note_temp:1.0 } },
  { name: 'Bach Prelude',   category: 'Composer',
    description: 'Continuous flowing motion — bright arpeggiation, harmonically rich.',
    params: { n_bars:24, bpm:100, chord_temp:1.0,  note_temp:1.0,  max_chord_notes:5, max_repeat:3,
              dur_bias:0.2 } },
  { name: 'Handel',         category: 'Composer',
    description: 'Stately Baroque rondo — ceremonial grandeur, clear rhythm, natural authority.',
    params: { n_bars:42, bpm:112, chord_temp:0.9,  note_temp:1.0,  form:'rondo',  max_chord_notes:4, max_repeat:3, b_note_temp:1.0,  top_k:38, top_p:0.88 } },
  { name: 'Vivaldi',        category: 'Composer',
    description: 'Brilliant and joyful arch — sparkling energy from first note to last.',
    params: { n_bars:32, bpm:120, chord_temp:1.0,  note_temp:1.0,  form:'arch',   max_chord_notes:4, max_repeat:2, b_note_temp:1.05 } },
  { name: 'Mozart',         category: 'Composer',
    description: 'Crisp classical rondo — symmetrical phrases, natural wit, elegant restraint.',
    params: { n_bars:42, bpm:120, chord_temp:0.9,  note_temp:0.95, form:'rondo',  max_chord_notes:4, max_repeat:2, b_note_temp:1.0,  top_p:0.88 } },
  { name: 'Beethoven',      category: 'Composer',
    description: 'Bold, decisive ABA — drama, full texture, triumphant resolution.',
    params: { n_bars:48, bpm:108, chord_temp:0.88, note_temp:1.0,  form:'aba',    max_chord_notes:5, max_repeat:3, b_note_temp:1.1,  top_p:0.88,
              dur_bias:0.2, bass_beat_strength:0.5 } },
  { name: 'Chopin Nocturne',category: 'Composer',
    description: 'Slow, melancholic ABA — intimate melody, warm harmonies, freer middle section.',
    params: { n_bars:32, bpm:58,  chord_temp:0.88, note_temp:0.9,  form:'aba',    max_chord_notes:4, max_repeat:3, b_note_temp:1.0,  top_k:38, top_p:0.88,
              dur_bias:0.7, melody_strength:0.5 } },
  { name: 'Chopin Étude',   category: 'Composer',
    description: 'Fast, driven binary — technical and expressive, A intense, B breathing.',
    params: { n_bars:32, bpm:120, chord_temp:0.95, note_temp:1.0,  form:'binary', max_chord_notes:4, max_repeat:2, b_note_temp:1.05, top_k:38, top_p:0.88 } },
  { name: 'Schubert',       category: 'Composer',
    description: 'Lyrical ABA — warm melody, romantic harmonies, chromatic wandering in the B.',
    params: { n_bars:40, bpm:88,  chord_temp:0.95, note_temp:0.95, form:'aba',    max_chord_notes:5, max_repeat:3, b_note_temp:1.05, top_k:38, top_p:0.88,
              dur_bias:0.4, melody_strength:0.4 } },
  { name: 'Brahms',         category: 'Composer',
    description: 'Heavy, warm ABA — serious, dense, harmonically rich, bass-grounded.',
    params: { n_bars:48, bpm:84,  chord_temp:0.88, note_temp:0.95, form:'aba',    max_chord_notes:5, max_repeat:3, b_note_temp:1.05, top_p:0.88,
              dur_bias:0.4, melody_strength:0.3, bass_beat_strength:0.4 } },
];

// ── State ──────────────────────────────────────────────────────────────────
let selectedPreset  = null;
let currentJobId    = null;
let pollInterval    = null;

let audioCtx         = null;
let masterGain       = null;
let pendingVolume    = null;
let instrument       = null;
let isPlaying        = false;
let playStartAcTime  = 0;
let pausePosition    = 0;
let scheduledSources = [];
let scheduleTimerId  = null;
let scheduledUpTo    = 0;       // song-time (secs) scheduled so far
let songDuration     = 0;
let noteData         = null;
let animFrameId      = null;
let currentSongFile  = null;

const SCHEDULE_LOOKAHEAD = 0.4;  // seconds ahead to schedule
const SCHEDULE_INTERVAL  = 150;  // ms between scheduler ticks

// ── DOM refs ───────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const presetSelect    = $('presetSelect');
const presetDescEl    = $('presetDescDisplay');
const generateBtn     = $('generateBtn');
const generationLog   = $('generationLog');
const logStatus       = $('logStatus');
const logSpinner      = $('logSpinner');
const playerSection   = $('playerSection');
const nowPlayingName  = $('nowPlayingName');
const instrLoading    = $('instrumentLoading');
const playBtn         = $('playBtn');
const progressFill    = $('progressFill');
const playheadEl      = $('playhead');
const timeDisplay     = $('timeDisplay');
const downloadBtn     = $('downloadBtn');
const pianoRoll       = $('pianoRoll');
const songList        = $('songList');
const rc              = pianoRoll.getContext('2d');

// ── Preset dropdown ────────────────────────────────────────────────────────
function populateSelect() {
  const groups = ['Base', 'Structured', 'Character', 'Composer'];
  groups.forEach(cat => {
    const grp = document.createElement('optgroup');
    grp.label = cat;
    PRESETS.filter(p => p.category === cat).forEach((p, i) => {
      const opt = document.createElement('option');
      opt.value = PRESETS.indexOf(p);
      opt.textContent = p.name;
      grp.appendChild(opt);
    });
    presetSelect.appendChild(grp);
  });
}

const bpmDisplay  = $('bpmDisplay');
const bpmOverride = $('bpmOverride');

bpmOverride.addEventListener('change', () => {
  if (bpmOverride.checked) {
    bpmDisplay.removeAttribute('readonly');
    bpmDisplay.focus();
    bpmDisplay.select();
  } else {
    bpmDisplay.setAttribute('readonly', '');
    if (selectedPreset) bpmDisplay.value = selectedPreset.params.bpm ?? '';
  }
});

function selectPreset(preset) {
  selectedPreset = preset;
  presetDescEl.textContent = preset.description;
  generateBtn.disabled = false;
  if (!bpmOverride.checked) bpmDisplay.value = preset.params.bpm ?? '';
}

function onPresetChange() {
  const idx = parseInt(presetSelect.value, 10);
  if (!isNaN(idx) && PRESETS[idx]) selectPreset(PRESETS[idx]);
}
presetSelect.addEventListener('change', onPresetChange);
presetSelect.addEventListener('input',  onPresetChange);

// ── Audio ──────────────────────────────────────────────────────────────────
async function ensureAudioCtx() {
  if (!audioCtx) {
    audioCtx   = new (window.AudioContext || window.webkitAudioContext)();
    masterGain = audioCtx.createGain();
    masterGain.gain.value = (pendingVolume !== null && pendingVolume !== undefined) ? pendingVolume : $('volumeSlider').value / 100;

    // Limiter/compressor to prevent clipping when many notes sound simultaneously
    const comp = audioCtx.createDynamicsCompressor();
    comp.threshold.value = -14;   // start compressing at -14 dBFS
    comp.knee.value      = 6;
    comp.ratio.value     = 8;
    comp.attack.value    = 0.003;
    comp.release.value   = 0.20;
    masterGain.connect(comp);
    comp.connect(audioCtx.destination);
  }
  if (audioCtx.state === 'suspended') await audioCtx.resume();
  return audioCtx;
}

async function ensureInstrument() {
  if (instrument) return instrument;
  instrLoading.hidden = false;
  const ac = await ensureAudioCtx();

  try {
    const sfLoad = Soundfont.instrument(ac, 'acoustic_grand_piano', {
      from: '/static/soundfonts/',
      format: 'mp3',
      destination: masterGain,
    });
    const timeout = new Promise((_, rej) =>
      setTimeout(() => rej(new Error('timeout')), 10000));
    instrument = await Promise.race([sfLoad, timeout]);
  } catch (e) {
    console.warn('Soundfont unavailable, using built-in synth:', e.message);
    instrument = 'synth';
  }

  instrLoading.hidden = true;
  return instrument;
}

function synthNote(pitch, when, duration, gain) {
  const freq    = 440 * Math.pow(2, (pitch - 69) / 12);
  const env     = audioCtx.createGain();
  env.connect(masterGain);

  const oscs = [[1,'triangle',0.55],[2,'sine',0.2],[3,'sine',0.08]].map(([mult,type,amp]) => {
    const osc = audioCtx.createOscillator();
    const g   = audioCtx.createGain();
    osc.type = type;
    osc.frequency.value = freq * mult;
    g.gain.value = amp;
    osc.connect(g); g.connect(env);
    osc.start(when); osc.stop(when + duration + 0.6);
    return osc;
  });

  const rel = Math.min(0.5, duration * 0.25 + 0.12);
  env.gain.setValueAtTime(0, when);
  env.gain.linearRampToValueAtTime(gain, when + 0.006);
  env.gain.exponentialRampToValueAtTime(gain * 0.55, when + 0.08);
  env.gain.setValueAtTime(gain * 0.55, when + duration);
  env.gain.exponentialRampToValueAtTime(0.0001, when + duration + rel);

  return {
    stop() {
      const t = audioCtx.currentTime;
      env.gain.cancelScheduledValues(t);
      env.gain.setValueAtTime(env.gain.value, t);
      env.gain.exponentialRampToValueAtTime(0.0001, t + 0.04);
      oscs.forEach(o => { try { o.stop(t + 0.05); } catch (_) {} });
    }
  };
}

// ── Piano roll ─────────────────────────────────────────────────────────────
function sizePianoRoll() {
  pianoRoll.width  = pianoRoll.offsetWidth || 800;
  pianoRoll.height = 190;
}

function drawRoll(currentTime) {
  const W = pianoRoll.width, H = pianoRoll.height;
  rc.clearRect(0, 0, W, H);

  // Background
  rc.fillStyle = '#03030a';
  rc.fillRect(0, 0, W, H);

  // Faint octave lines (C notes: 24,36,48,60,72,84,96)
  rc.fillStyle = 'rgba(201,160,85,0.07)';
  for (let pitch = 24; pitch <= 108; pitch += 12) {
    const y = H - ((pitch - 21) / 87) * H;
    rc.fillRect(0, y, W, 1);
  }

  if (!noteData || !songDuration) return;

  // Notes
  noteData.notes.forEach(n => {
    const x  = (n.start / songDuration) * W;
    const w  = Math.max(2, ((n.end - n.start) / songDuration) * W);
    const y  = H - ((n.pitch - 21) / 87) * H;
    const h  = Math.max(2, H / 88);
    const a  = 0.35 + (n.vel / 127) * 0.65;
    rc.fillStyle = `rgba(201,160,85,${a.toFixed(2)})`;
    rc.fillRect(x, y - h * 0.5, w, h);
  });

  // Playhead
  if (currentTime > 0) {
    const px = (currentTime / songDuration) * W;
    rc.fillStyle = 'rgba(255,255,255,0.75)';
    rc.fillRect(px - 0.75, 0, 1.5, H);
  }
}

function updateProgress(currentTime) {
  const pct = songDuration > 0 ? Math.min(100, (currentTime / songDuration) * 100) : 0;
  progressFill.style.width = pct + '%';
  playheadEl.style.left    = pct + '%';
  timeDisplay.textContent  = `${fmt(currentTime)} / ${fmt(songDuration)}`;
  drawRoll(currentTime);
}

function animLoop() {
  if (!isPlaying) return;
  const elapsed = audioCtx.currentTime - playStartAcTime;
  const t = Math.min(elapsed, songDuration);
  updateProgress(t);
  if (t < songDuration) {
    animFrameId = requestAnimationFrame(animLoop);
  } else {
    stopPlayback();
    pausePosition = 0;
    updateProgress(0);
  }
}

function stopScheduled() {
  scheduledSources.forEach(src => { try { src.stop(); } catch (_) {} });
  scheduledSources = [];
}

function stopPlayback() {
  isPlaying = false;
  playBtn.textContent = '▶';
  if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
  stopScheduler();
  stopScheduled();
}

function scheduleWindow() {
  if (!noteData || !instrument || !isPlaying) return;
  const ac       = audioCtx;
  const now      = ac.currentTime;
  const songNow  = now - playStartAcTime;          // current song position
  const through  = songNow + SCHEDULE_LOOKAHEAD;   // schedule up to here

  noteData.notes.forEach(n => {
    if (n.start < scheduledUpTo) return;           // already scheduled
    if (n.start > through) return;                 // too far ahead
    const when = playStartAcTime + n.start;
    if (when < now - 0.05) return;                 // already in the past
    const dur  = Math.max(0.05, n.end - n.start);
    const gain = 0.10 + (n.vel / 127) * 0.35;
    const src  = instrument === 'synth'
      ? synthNote(n.pitch, when, dur, gain)
      : instrument.play(n.pitch, when, { duration: dur, gain });
    if (src) scheduledSources.push(src);
  });

  scheduledUpTo = through;
}

function startScheduler(fromSongTime) {
  scheduledUpTo = fromSongTime;
  scheduleWindow();
  scheduleTimerId = setInterval(scheduleWindow, SCHEDULE_INTERVAL);
}

function stopScheduler() {
  if (scheduleTimerId !== null) {
    clearInterval(scheduleTimerId);
    scheduleTimerId = null;
  }
}

// ── MIDI loading ───────────────────────────────────────────────────────────
async function loadMidi(filename) {
  stopPlayback();
  pausePosition = 0;

  const nr     = await fetch(`/api/notes/${filename}`);
  noteData     = await nr.json();
  songDuration = noteData.duration || 0;

  currentSongFile = filename;
  nowPlayingName.textContent = prettyName(filename);
  downloadBtn.href = `/songs/${filename}`;
  downloadBtn.setAttribute('download', filename);
  playerSection.hidden = false;
  playerSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  sizePianoRoll();
  updateProgress(0);

  document.querySelectorAll('.song-item').forEach(el => {
    el.classList.toggle('playing', el.dataset.filename === filename);
  });
}

// ── Playback controls ──────────────────────────────────────────────────────
playBtn.addEventListener('click', async () => {
  if (!noteData) return;
  await ensureAudioCtx();
  if (isPlaying) {
    pausePosition = Math.min(audioCtx.currentTime - playStartAcTime, songDuration);
    stopPlayback();
  } else {
    await ensureInstrument();
    playStartAcTime = audioCtx.currentTime - pausePosition;
    isPlaying = true;
    playBtn.textContent = '⏸';
    startScheduler(pausePosition);
    animFrameId = requestAnimationFrame(animLoop);
  }
});

// Seek on click
$('progressTrack').addEventListener('click', async e => {
  if (!noteData || !songDuration) return;
  const rect       = e.currentTarget.getBoundingClientRect();
  const pct        = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  const secs       = pct * songDuration;
  const wasPlaying = isPlaying;
  stopPlayback();
  pausePosition = secs;
  await ensureAudioCtx();
  playStartAcTime = audioCtx.currentTime - secs;
  updateProgress(secs);
  if (wasPlaying) {
    isPlaying = true;
    playBtn.textContent = '⏸';
    startScheduler(secs);
    animFrameId = requestAnimationFrame(animLoop);
  }
});

window.addEventListener('resize', () => { sizePianoRoll(); drawRoll(0); });

// ── Volume ─────────────────────────────────────────────────────────────────
const volSlider = $('volumeSlider');
function updateVolTrack() {
  volSlider.style.setProperty('--vol-pct', volSlider.value + '%');
}
volSlider.addEventListener('input', () => {
  if (masterGain) masterGain.gain.value = volSlider.value / 100;
  else pendingVolume = volSlider.value / 100;
  updateVolTrack();
});
updateVolTrack();

// ── Generation ─────────────────────────────────────────────────────────────
function pollStatus(job_id, attempt) {
  attempt = attempt || 0;
  if (attempt > 300) { // ~10 minutes max
    logStatus.textContent = '✗ Timed out waiting for generation';
    logSpinner.style.display = 'none';
    resetGenerateBtn();
    return;
  }
  fetch(`/api/status/${job_id}`)
    .then(function(r) { return r.json(); })
    .then(async function(s) {
      if (s.status === 'done') {
        logStatus.textContent = '✓ Generated';
        logSpinner.style.display = 'none';
        resetGenerateBtn();
        try { await loadMidi(s.filename); } catch (e) { console.error('loadMidi error:', e); }
        try { await refreshLibrary(); } catch (e) { console.error('refreshLibrary error:', e); }
      } else if (s.status === 'error') {
        logStatus.textContent = '✗ Generation failed';
        logSpinner.style.display = 'none';
        resetGenerateBtn();
      } else {
        setTimeout(function() { pollStatus(job_id, attempt + 1); }, 2000);
      }
    })
    .catch(function() {
      // network hiccup — retry
      setTimeout(function() { pollStatus(job_id, attempt + 1); }, 3000);
    });
}

generateBtn.addEventListener('click', async () => {
  if (!selectedPreset) {
    generationLog.hidden = false;
    logStatus.textContent = '⚠ Please select a preset first';
    logSpinner.style.display = 'none';
    return;
  }

  generateBtn.disabled = true;
  generateBtn.classList.add('generating');
  generateBtn.querySelector('.btn-label').textContent = 'Generating…';

  generationLog.hidden = false;
  logStatus.textContent = 'Generating…';
  logSpinner.style.display = '';

  try {
    const params = { ...BASE_PARAMS, ...selectedPreset.params, raw: true };
    if (bpmOverride.checked) {
      const customBpm = parseInt(bpmDisplay.value, 10);
      if (customBpm >= 20 && customBpm <= 300) params.bpm = customBpm;
    }
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: selectedPreset.name, params }),
    });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();
    const job_id = data.job_id;
    if (!job_id) throw new Error('No job ID returned');
    currentJobId = job_id;
    pollStatus(job_id);

  } catch (err) {
    logStatus.textContent = `✗ Request failed: ${err.message}`;
    logSpinner.style.display = 'none';
    resetGenerateBtn();
  }
});

function resetGenerateBtn() {
  generateBtn.disabled = false;
  generateBtn.classList.remove('generating');
  generateBtn.querySelector('.btn-label').textContent = 'Generate';
}

// ── Library ────────────────────────────────────────────────────────────────
async function refreshLibrary() {
  const songs = await fetch('/api/songs').then(r => r.json());
  if (!songs.length) {
    songList.innerHTML = '<div class="empty-state">No songs yet — generate something above.</div>';
    return;
  }
  songList.innerHTML = '';
  songs.forEach(song => {
    const item = document.createElement('div');
    item.className = 'song-item' + (song.filename === currentSongFile ? ' playing' : '');
    item.dataset.filename = song.filename;
    const kb = (song.size / 1024).toFixed(1);
    item.innerHTML = `
      <button class="song-play-btn" title="Play">▶</button>
      <span class="song-name">${prettyName(song.filename)}</span>
      <span class="song-meta">${kb} KB</span>
      <div class="song-actions">
        <button class="song-action-btn rename-btn" title="Rename">✎</button>
        <button class="song-action-btn delete-btn" title="Delete">✕</button>
      </div>
      <a class="song-dl" href="/songs/${song.filename}" download="${song.filename}">↓</a>
    `;

    item.querySelector('.song-play-btn').addEventListener('click', () => loadMidi(song.filename));

    item.querySelector('.rename-btn').addEventListener('click', () => {
      startRename(item, song.filename);
    });

    item.querySelector('.delete-btn').addEventListener('click', async () => {
      if (!confirm(`Delete "${prettyName(song.filename)}"?`)) return;
      await fetch(`/api/songs/${song.filename}`, { method: 'DELETE' });
      if (currentSongFile === song.filename) {
        stopPlayback();
        playerSection.hidden = true;
        currentSongFile = null;
      }
      await refreshLibrary();
    });

    songList.appendChild(item);
  });
}

function startRename(item, filename) {
  const nameEl  = item.querySelector('.song-name');
  const current = prettyName(filename);
  nameEl.innerHTML = `
    <input class="song-rename-input" value="${current}" maxlength="80">
    <button class="rename-save" title="Save">✓</button>
    <button class="rename-cancel" title="Cancel">✗</button>
  `;
  const input = nameEl.querySelector('.song-rename-input');
  input.focus();
  input.select();

  async function doRename() {
    const val = input.value.trim();
    if (!val || val === current) { nameEl.textContent = current; return; }
    const res  = await fetch(`/api/songs/${filename}/rename`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ new_name: val }),
    });
    const data = await res.json();
    if (data.filename && currentSongFile === filename) {
      currentSongFile = data.filename;
      nowPlayingName.textContent = prettyName(data.filename);
      downloadBtn.href = `/songs/${data.filename}`;
      downloadBtn.setAttribute('download', data.filename);
    }
    await refreshLibrary();
  }

  nameEl.querySelector('.rename-save').addEventListener('click', doRename);
  nameEl.querySelector('.rename-cancel').addEventListener('click', () => {
    nameEl.textContent = current;
  });
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') doRename();
    if (e.key === 'Escape') nameEl.textContent = current;
  });
}

$('refreshBtn').addEventListener('click', refreshLibrary);

// ── Helpers ────────────────────────────────────────────────────────────────
function prettyName(filename) {
  return filename
    .replace(/_\d{8}_\d{6}\.mid$/, '')
    .replace(/_+/g, ' ')
    .trim()
    .replace(/\b\w/g, c => c.toUpperCase());
}

function fmt(s) {
  const m   = Math.floor(s / 60);
  const sec = Math.floor(s % 60).toString().padStart(2, '0');
  return `${m}:${sec}`;
}

// ── Init ───────────────────────────────────────────────────────────────────
populateSelect();
onPresetChange(); // sync if browser restored a previous selection
refreshLibrary();
sizePianoRoll();
drawRoll(0);
