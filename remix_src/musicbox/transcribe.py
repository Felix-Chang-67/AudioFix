"""Full vocal-melody transcription on the source 16th grid -> notes json."""
import sys, json, numpy as np, soundfile as sf, librosa
stems, pitchfile, out = sys.argv[1], sys.argv[2], sys.argv[3]
BPM, PH = 145.01, 0.015; BAR = 240/BPM; S16 = BAR/16
SR16 = 16000; HOP = 80
x, sr = sf.read(f"{stems}/04_shutter_doll_vocals.wav", dtype='float32'); v = x.mean(1)
y = librosa.resample(v, orig_sr=sr, target_sr=SR16)
z = np.load(pitchfile); midi, vf, vp, ft = z['midi'], z['vf'], z['vp'], z['ft']
f0 = librosa.midi_to_hz(np.where(vf, midi, 0))
# --- octave check by harmonic salience
X = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP, center=True))
nf = min(X.shape[1], len(f0)); X = X[:, :nf]; f0 = f0[:nf]; vf = vf[:nf]; ft = ft[:nf]; vp = vp[:nf]
binhz = SR16/2048
def sal(f, col):
    s = 0.0
    for h in range(1, 11):
        b = h*f/binhz
        if b >= X.shape[0]-2: break
        i = int(round(b)); s += (0.84**h) * X[i-1:i+2, col].max()
    return s
fixed = np.zeros(nf); nup = ndn = 0
for i in range(nf):
    if not vf[i]: continue
    c = {k: f0[i]*k for k in (0.5, 1.0, 2.0) if 98 <= f0[i]*k <= 1100}
    sc = {k: sal(f, i) for k, f in c.items()}
    best = max(sc, key=sc.get)
    # require a clear margin before overriding pYIN
    if best != 1.0 and sc[best] > 1.25*sc.get(1.0, 0):
        fixed[i] = f0[i]*best; nup += best == 2.0; ndn += best == 0.5
    else:
        fixed[i] = f0[i]
print(f"octave corrections: up {nup}, down {ndn} of {vf.sum()} voiced frames")
m = np.where(fixed > 0, librosa.hz_to_midi(np.maximum(fixed, 1)), np.nan)
TUNE = 0.10
m = m - TUNE
# --- onsets (syllables)
oenv = librosa.onset.onset_strength(y=y, sr=SR16, hop_length=HOP)
on = librosa.onset.onset_detect(onset_envelope=oenv, sr=SR16, hop_length=HOP, units='time', delta=0.12, wait=8)
rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0][:nf]
nslots = int((len(y)/SR16 - PH)/S16)
P = np.full(nslots, np.nan); V = np.zeros(nslots); O = np.zeros(nslots, bool); E = np.zeros(nslots)
for k in range(nslots):
    t0 = PH + k*S16
    sel = (ft >= t0 + 0.3*S16) & (ft < t0 + S16)
    allsel = (ft >= t0) & (ft < t0 + S16)
    vv = sel & ~np.isnan(m)
    V[k] = vv.sum()/max(1, sel.sum()); E[k] = rms[allsel].mean() if allsel.any() else 0
    if V[k] >= 0.5: P[k] = np.median(m[vv])
    O[k] = np.any((on >= t0 - 0.03) & (on < t0 + 0.6*S16))
Edb = 20*np.log10(E + 1e-9)
SCALE = {0, 2, 3, 5, 7, 8, 10}
def snap(p, stable):
    r = int(round(p))
    if r % 12 in SCALE or (r % 12 == 11 and stable): return r
    lo, hi = int(np.floor(p)), int(np.ceil(p))
    cands = [c for c in (r-1, r+1) if c % 12 in SCALE]
    return min(cands, key=lambda c: abs(c - p))
notes = []; cur = None
for k in range(nslots):
    p = P[k]
    if np.isnan(p) or Edb[k] < -45:
        cur = None; continue
    if cur is not None and abs(p - cur['pf'][-1]) < 0.7 and not (O[k] and k - cur['k'] >= 1 and abs(p - np.mean(cur['pf'])) < 0.7 and False):
        # continuation unless a syllable onset starts here
        if O[k]:
            cur = None
        else:
            cur['n'] += 1; cur['pf'].append(p); continue
    cur = {'k': k, 'n': 1, 'pf': [p]}
    notes.append(cur)
out_notes = []
for nt in notes:
    pf = np.array(nt['pf']); stable = nt['n'] >= 3 and np.ptp(pf) < 0.5
    out_notes.append({'slot': nt['k'], 'len': nt['n'], 'midi': snap(float(np.median(pf)), stable),
                      'pf': round(float(np.median(pf)), 2)})
# remove 1-slot glide notes that sit between neighbours (scoops / slides)
clean = []
for i, nt in enumerate(out_notes):
    if nt['len'] == 1 and 0 < i < len(out_notes)-1:
        a, b = out_notes[i-1], out_notes[i+1]
        if a['slot']+a['len'] == nt['slot'] and nt['slot']+1 == b['slot'] and min(a['midi'], b['midi']) < nt['midi'] < max(a['midi'], b['midi']) and abs(nt['pf'] - round(nt['pf'])) > 0.25:
            continue
    clean.append(nt)
json.dump(clean, open(out, 'w'))
print(f"{len(clean)} notes (from {len(out_notes)})")
PC = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
def name(mm): return f"{PC[mm%12]}{mm//12-1}"
def show(b0, b1):
    for b in range(b0, b1):
        row = ['  .  ']*16
        for nt in clean:
            if b*16 <= nt['slot'] < (b+1)*16:
                s = nt['slot'] - b*16; row[s] = f"{name(nt['midi']):>4}" + ('' if nt['len'] == 1 else '-')
                for j in range(1, nt['len']):
                    if s+j < 16 and row[s+j] == '  .  ': row[s+j] = '  ~  '
        print(f"{b:3d} " + "".join(f"{r:5s}" for r in row))
for a, b in [(16, 24), (32, 46), (70, 84), (108, 122)]:
    print(f"--- bars {a}-{b-1}"); show(a, b)
