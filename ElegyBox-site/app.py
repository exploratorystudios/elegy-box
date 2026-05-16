import os, re, sys, uuid, time, threading
from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory, request, abort
import mido

ELEGY_DIR = Path(__file__).resolve().parent.parent
SONGS_DIR = Path(__file__).parent / 'songs'
SONGS_DIR.mkdir(exist_ok=True)

PYTHON = str(Path(sys.executable))   # Chesswork venv python

app = Flask(__name__)
jobs = {}  # job_id -> {status, lines, filename}


def _run(job_id, cmd_args, out_path):
    import subprocess
    jobs[job_id].update(status='running', lines=[], filename=None)
    try:
        cmd = [PYTHON, str(ELEGY_DIR / 'generate.py')] + cmd_args + ['--output', str(out_path)]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(ELEGY_DIR)
        )
        for line in proc.stdout:
            jobs[job_id]['lines'].append(line.rstrip())
        proc.wait()
        if proc.returncode == 0 and out_path.exists():
            jobs[job_id].update(status='done', filename=out_path.name)
        else:
            jobs[job_id]['status'] = 'error'
    except Exception as e:
        jobs[job_id].update(status='error', lines=jobs[job_id]['lines'] + [str(e)])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate():
    data   = request.get_json()
    raw    = data.get('name', 'piece')
    slug   = re.sub(r'[^a-z0-9]+', '_', raw.lower()).strip('_')
    ts     = time.strftime('%Y%m%d_%H%M%S')
    out    = SONGS_DIR / f'{slug}_{ts}.mid'

    params = data.get('params', {})
    args   = []
    for k, v in params.items():
        if v is True:
            args.append(f'--{k}')          # store_true flag — no value
        elif v is not False:               # False means omit the flag entirely
            args.extend([f'--{k}', str(v)])

    jid = str(uuid.uuid4())
    jobs[jid] = {'status': 'pending', 'lines': [], 'filename': None}
    threading.Thread(target=_run, args=(jid, args, out), daemon=True).start()
    return jsonify({'job_id': jid})


@app.route('/api/status/<jid>')
def job_status(jid):
    j = jobs.get(jid)
    if not j:
        abort(404)
    return jsonify(j)


@app.route('/api/songs')
def list_songs():
    songs = []
    for f in sorted(SONGS_DIR.iterdir(), reverse=True):
        if f.suffix == '.mid':
            songs.append({'filename': f.name, 'size': f.stat().st_size})
    return jsonify(songs)


@app.route('/api/notes/<filename>')
def get_notes(filename):
    path = SONGS_DIR / filename
    if not path.exists():
        abort(404)
    try:
        mid  = mido.MidiFile(str(path))
        tpb  = mid.ticks_per_beat
        tempo = 500000
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    break
        spt = tempo / 1e6 / tpb  # seconds per tick

        active, notes = {}, []
        for track in mid.tracks:
            t = 0
            for msg in track:
                t += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    active[msg.note] = (t, msg.velocity)
                elif (msg.type == 'note_off' or
                        (msg.type == 'note_on' and msg.velocity == 0)) and msg.note in active:
                    s, vel = active.pop(msg.note)
                    notes.append({
                        'pitch': msg.note,
                        'start': round(s * spt, 4),
                        'end':   round(t * spt, 4),
                        'vel':   vel,
                    })

        dur = max((n['end'] for n in notes), default=0)
        return jsonify({'notes': notes, 'duration': round(dur, 3)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/songs/<filename>', methods=['DELETE'])
def delete_song(filename):
    path = SONGS_DIR / filename
    if not path.exists():
        abort(404)
    path.unlink()
    return jsonify({'ok': True})


@app.route('/api/songs/<filename>/rename', methods=['POST'])
def rename_song(filename):
    path = SONGS_DIR / filename
    if not path.exists():
        abort(404)
    new_name = (request.get_json() or {}).get('new_name', '').strip()
    if not new_name:
        return jsonify({'error': 'empty name'}), 400
    m = re.search(r'(_\d{8}_\d{6})\.mid$', filename)
    ts   = m.group(1) if m else ''
    slug = re.sub(r'[^a-z0-9]+', '_', new_name.lower()).strip('_') or 'piece'
    new_filename = f'{slug}{ts}.mid'
    path.rename(SONGS_DIR / new_filename)
    return jsonify({'filename': new_filename})


@app.route('/songs/<filename>')
def serve_song(filename):
    return send_from_directory(SONGS_DIR, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5173, threaded=True)
