"""Shutter Doll (Afterimage) — piano & strings rearrangement.

usage: python3 -I rearrange.py <scripts_dir> <stems_dir> <melody.json> <chorus_voted.json> <out_dir>

Only the lead vocal stem is used. Everything else is newly written MIDI (piano, string section,
cello, contrabass, pizzicato, harp, timpani, cymbal) rendered with FluidSynth + FluidR3_GM.
"""
import sys
import json
import subprocess
import numpy as np
import soundfile as sf
import scipy.signal as ss
import pretty_midi as pm
import pedalboard as pb

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from remixlib.dsp import SR, db, butter, reverb, compress, limiter, true_peak_db, board, delay
import arr_chart as A

STEMS, MELFILE, VOTEDFILE, OUT = sys.argv[2:6]
SF2 = '/usr/share/sounds/sf2/FluidR3_GM.sf2'
RNG = np.random.default_rng(7)

BPM = 145.01
BEAT = 60 / BPM
BAR = 4 * BEAT
SRC_PH = 0.015
T0 = 1.0
NBARS = 83

# ---------------------------------------------------------------- time map (rubato only where there is no voice)
RIT = {7: 0.18, 79: 0.08, 80: 0.16, 81: 0.32, 82: 0.6}
bar_dur = np.array([BAR * (1 + RIT.get(k, 0)) for k in range(NBARS + 1)])
bar_start = T0 + np.concatenate([[0], np.cumsum(bar_dur)])


def tm(ob):
    k = int(np.floor(ob))
    u = ob - k
    r = RIT.get(k, 0)
    return bar_start[k] + BAR * (u + r * u * u)


def ST(sb):
    return SRC_PH + sb * BAR


def IDX(t):
    return int(round(t * SR))


# ---------------------------------------------------------------- melody knowledge (for voicing decisions)
mel = json.load(open(MELFILE))
for nt in mel:
    while nt['midi'] < 55:
        nt['midi'] += 12
    while nt['midi'] > 76:
        nt['midi'] -= 12
HELD = {}   # half-bar index -> set of pitch classes held there by the lead line


def mark_held(ob0, ob1, pcl):
    for hb in range(int(ob0 * 2), int(np.ceil(ob1 * 2 - 1e-9))):
        HELD.setdefault(hb, set()).add(pcl % 12)


for a, b, off, semi in A.VOCAL_SEGMENTS:
    for nt in mel:
        sb = nt['slot'] / 16
        if a <= sb < b and nt['len'] >= 2:
            mark_held(sb - off, sb - off + nt['len'] / 16, nt['midi'] + semi)

# instrumental lead lines (out-bar positions, midi in C-minor terms)
HOOK = [(0, 79), (4, 87), (12, 87), (16, 87), (18, 86), (24, 86), (28, 82), (30, 79), (32, 79), (36, 80), (38, 82),
        (40, 80), (41, 79), (43, 80), (45, 75), (47, 74), (48, 75), (50, 74), (52, 75), (56, 79), (58, 72)]
INTRO_MEL = []
for i, (s, m) in enumerate(HOOK):
    nxt = HOOK[i + 1][0] if i + 1 < len(HOOK) else 64
    INTRO_MEL.append((s / 8, m, min(1.0, (nxt - s) / 8)))
voted = json.load(open(VOTEDFILE))
STR_MEL = []
for nt in voted:
    if nt['slot'] >= 7 * 16 + 8:
        break
    if nt['len'] >= 2 or nt['slot'] % 2 == 0:
        if STR_MEL and STR_MEL[-1][1] == nt['midi'] and nt['slot'] / 16 - STR_MEL[-1][0] < 0.25:
            continue
        STR_MEL.append([nt['slot'] / 16, nt['midi'], 0])
for i in range(len(STR_MEL)):
    end = STR_MEL[i + 1][0] if i + 1 < len(STR_MEL) else 7.5
    STR_MEL[i][2] = min(1.5, end - STR_MEL[i][0])
OUTRO_MEL = [(79, 79, 0.25), (79.25, 87, 0.5), (79.75, 87, 0.25), (80, 87, 0.125), (80.125, 86, 0.375),
             (80.5, 86, 0.25), (80.75, 82, 0.125), (80.875, 79, 0.125), (81, 79, 0.375), (81.375, 77, 0.125),
             (81.5, 74, 0.5), (82, 76, 2.0)]
for s, m, d in INTRO_MEL:
    mark_held(s, s + d, m)
for s, m, d in STR_MEL:
    mark_held(51 + s, 51 + s + d, m)
for s, m, d in OUTRO_MEL:
    mark_held(s, s + d, m + 1)


def chord(ob):
    sym, (root, bass, pcs) = A.chord_at(ob)
    return sym, root, bass, pcs


def avoid(pcs, ob, keep=()):
    held = HELD.get(int(ob * 2), set())
    out = [p for p in pcs if p in keep or all(min((p - h) % 12, (h - p) % 12) != 1 for h in held)]
    return out or pcs[:1]


def inst_range(pcl, lo, hi):
    return [m for m in range(lo, hi + 1) if m % 12 == pcl]


def nearest(pcl, target, lo, hi):
    c = inst_range(pcl, lo, hi)
    return min(c, key=lambda m: abs(m - target)) if c else None


def voicing(pcs, lo, hi, prev=None, n=4, center=None):
    """Voice-led close voicing of up to n notes from pcs within [lo, hi]."""
    pcs = list(dict.fromkeys(pcs))
    cands = []
    for rot in range(len(pcs)):
        order = pcs[rot:] + pcs[:rot]
        start = inst_range(order[0], lo, lo + 11)
        if not start:
            continue
        v = [start[0]]
        i = 1
        while len(v) < n:
            p = order[i % len(order)]
            m = v[-1] + ((p - v[-1]) % 12 or 12)
            v.append(m)
            i += 1
        if v[-1] <= hi:
            cands.append(v)
    if not cands:
        return []
    if prev:
        def cost(v):
            return sum(abs(a - b) for a, b in zip(sorted(v), sorted(prev)[:len(v)]))
    else:
        c0 = center or (lo + hi) / 2
        def cost(v):
            return abs(np.mean(v) - c0)
    return min(cands, key=cost)


def bass_of(ob, lo, hi):
    sym, root, bass, pcs = chord(ob)
    return nearest(bass, (lo + hi) / 2, lo, hi)


def section(ob):
    for name, a, b in A.SECTIONS:
        if a <= ob < b:
            return name
    return 'outro'


# ---------------------------------------------------------------- MIDI parts
piano = pm.Instrument(program=0, name='piano')
pad = pm.Instrument(program=49, name='strings pad')
trem = pm.Instrument(program=44, name='tremolo strings')
vln = pm.Instrument(program=48, name='violins')
vc = pm.Instrument(program=42, name='cellos')
cb = pm.Instrument(program=43, name='contrabass')
pizz = pm.Instrument(program=45, name='pizzicato')
harp = pm.Instrument(program=46, name='harp')
timp = pm.Instrument(program=47, name='timpani')
cym = pm.Instrument(program=0, is_drum=True, name='cymbals')   # standard kit (orchestra kit maps 41-53 to timpani)


def N(inst, m, ob0, ob1, vel, jitter=0.0, early=0.0):
    if m is None:
        return
    t0 = tm(ob0) + RNG.uniform(-jitter, jitter) - early
    t1 = tm(ob1) - early - 0.01
    if t1 <= t0 + 0.02:
        t1 = t0 + 0.05
    inst.notes.append(pm.Note(velocity=int(np.clip(vel + RNG.integers(-3, 4), 1, 127)), pitch=int(m),
                              start=max(0, t0), end=t1))


def CC(inst, num, ob, val):
    inst.control_changes.append(pm.ControlChange(number=num, value=int(np.clip(val, 0, 127)), time=tm(ob)))


def ramp(inst, ob0, ob1, v0, v1, num=11, steps=16):
    for k in range(steps + 1):
        u = k / steps
        CC(inst, num, ob0 + (ob1 - ob0) * u, v0 + (v1 - v0) * u)


def shift(ob):
    return 1 if ob >= A.KEY_UP_FROM else 0


# --- piano -----------------------------------------------------------------
def pedal(ob0, ob1):
    piano.control_changes.append(pm.ControlChange(number=64, value=127, time=tm(ob0) + 0.03))
    piano.control_changes.append(pm.ControlChange(number=64, value=0, time=tm(ob1) - 0.025))


def lh_arp(ob, vel, pattern=(0, 1, 2, 3, 2, 1, 2, 3), lo_b=36, hi_b=47, half_only=None):
    """Left-hand broken chord in 8ths over one bar (re-voiced per half bar)."""
    for k, idx in enumerate(pattern):
        pos = ob + k / 8
        if half_only is not None and int(k >= 4) != half_only:
            continue
        sym, root, bass, pcs = chord(pos)
        b = nearest(bass, (lo_b + hi_b) / 2, lo_b, hi_b)
        upper = voicing(avoid([p for p in pcs if p != bass] or pcs, pos), b + 7, b + 24, n=3, center=b + 13)
        tones = [b] + upper
        N(piano, tones[min(idx, len(tones) - 1)], pos, pos + 0.5, vel + (8 if k in (0, 4) else 0), jitter=0.006)


prev_rh = None
for ob in range(NBARS):
    sec = section(ob)
    for half in (0, 1):
        p0 = ob + half / 2
        sym, root, bass, pcs = chord(p0)
        same_as_prev = half == 1 and A.C[ob][0] == A.C[ob][1]
        if not same_as_prev:
            span = 1.0 if (half == 0 and A.C[ob][0] == A.C[ob][1]) else 0.5
            if sec != 'outro' or ob < 82:
                pedal(p0, p0 + span)
    if sec == 'intro':
        lh_arp(ob, 46 if ob < 7 else 40)
    elif sec == 'verse':
        b = bass_of(ob, 36, 47)
        if ob < 16:                  # the second half of the verse has a moving left hand instead
            N(piano, b, ob, ob + 1, 50, 0.006)
            N(piano, b + 12, ob, ob + 1, 38, 0.006)
        for half in (0, 1):
            p0 = ob + half / 2
            if half == 1 and A.C[ob][0] == A.C[ob][1] and ob < 16:
                continue
            sym, root, bass, pcs = chord(p0)
            v = voicing(avoid(pcs, p0), 67, 81, prev_rh, n=3)
            prev_rh = v
            for m in v:
                N(piano, m, p0 + (0.25 if ob < 16 else 0), p0 + 0.5, 40, 0.012)
        if ob >= 16:
            lh_arp(ob, 36, pattern=(0, 1, 2, 3, 2, 1, 2, 3))
    elif sec in ('chorus', 'final'):
        boost = 10 if sec == 'final' else 0
        for half in (0, 1):
            p0 = ob + half / 2
            b = bass_of(p0, 31, 43)
            if half == 0 or A.C[ob][0] != A.C[ob][1]:
                N(piano, b, p0, p0 + 0.5, 62 + boost, 0.004)
                N(piano, b + 12, p0, p0 + 0.5, 56 + boost, 0.004)
            else:
                N(piano, b + 12, p0, p0 + 0.5, 50 + boost, 0.004)
            sym, root, bass, pcs = chord(p0)
            v = voicing(avoid([p for p in pcs if p != bass] or pcs, p0), 55, 69, prev_rh, n=3)
            prev_rh = v
            for k in range(4):       # 8th-note pulse
                pos = p0 + k / 8
                for m in v:
                    N(piano, m, pos, pos + 0.11, (54 if k == 0 else 42) + boost, 0.005)
            if sec == 'final' and half == 0:   # top sparkle
                top = voicing(avoid(pcs, p0), 79, 93, None, n=2, center=86)
                for m in top:
                    N(piano, m, p0, p0 + 0.5, 46, 0.004)
    elif sec == 'turn':
        sym, root, bass, pcs = chord(ob)
        run = sorted({m for p in pcs for m in inst_range(p, 60, 86)}, reverse=True)
        for k, m in enumerate(run[:12]):
            N(piano, m, ob + k / 16, ob + k / 16 + 0.3, 50 - k, 0.004)
        N(piano, bass_of(ob, 36, 47), ob, ob + 1, 52)
    elif sec == 'bridge':
        b = bass_of(ob, 36, 47)
        N(piano, b, ob, ob + 1, 46, 0.006)
        for k in range(8):           # high right-hand arpeggio, 8ths
            pos = ob + k / 8
            sym, root, bass, pcs = chord(pos)
            v = voicing(avoid(pcs, pos), 72, 88, None, n=4, center=79)
            if v:
                N(piano, v[[0, 1, 2, 3, 2, 1, 2, 3][k] % len(v)], pos, pos + 0.25, 48 + (6 if k % 4 == 0 else 0), 0.006)
    elif sec == 'strings':
        lh_arp(ob, 44)
        for half in (0, 1):
            p0 = ob + half / 2
            sym, root, bass, pcs = chord(p0)
            v = voicing(avoid(pcs, p0), 72, 86, prev_rh, n=3)
            prev_rh = v
            for m in v:
                N(piano, m, p0, p0 + 0.5, 40, 0.008)
    elif sec == 'tag':
        lh_arp(ob, 44 - 2 * (ob - 73))
        sym, root, bass, pcs = chord(ob)
        for m in voicing(avoid(pcs, ob), 72, 86, prev_rh, n=3):
            N(piano, m, ob, ob + 1, 38, 0.008)
    elif sec == 'outro':
        if ob < 82:
            lh_arp(ob, 40 - 2 * (ob - 78))
        else:
            sym, root, bass, pcs = chord(ob)
            b = bass_of(ob, 36, 47)
            roll = [b, b + 7] + voicing(pcs, b + 12, b + 28, n=4, center=b + 19)
            for k, m in enumerate(roll):
                N(piano, m, ob + k * 0.04, ob + 2.6, 44 + 2 * k, 0)
            piano.control_changes.append(pm.ControlChange(number=64, value=127, time=tm(ob) - 0.01))
            piano.control_changes.append(pm.ControlChange(number=64, value=0, time=tm(ob) + 7.0))
# intro / outro melodies (right hand)
for s, m, d in INTRO_MEL:
    N(piano, m, s, s + d, 60 if s < 7 else 52, 0.008)
for s, m, d in OUTRO_MEL:
    N(piano, m + 1, s, s + d, 56 if s < 82 else 50, 0.008)

# --- strings ---------------------------------------------------------------
STR_EARLY = 0.035
prev_pad = None
PAD_LEVEL = {}   # section -> (cc11 start, end)
for ob in range(20, 82):
    sec = section(ob)
    if sec in ('turn',) or (sec == 'bridge' and ob >= 47):
        pass
    for half in (0, 1):
        p0 = ob + half / 2
        if half == 1 and A.C[ob][0] == A.C[ob][1]:
            continue
        span = 1.0 if (half == 0 and A.C[ob][0] == A.C[ob][1]) else 0.5
        sym, root, bass, pcs = chord(p0)
        v = voicing(avoid([p for p in pcs if p != bass] or pcs, p0), 55, 79, prev_pad, n=4)
        prev_pad = v
        target = trem if (sec == 'bridge' and ob >= 47) else pad
        if sec == 'outro' and ob >= 79:
            continue
        for m in v:
            N(target, m, p0, p0 + span + 0.02, 70, early=STR_EARLY)
# pad dynamics
ramp(pad, 19.9, 23.9, 20, 70)
ramp(pad, 24, 26, 80, 92)
ramp(pad, 36, 38.9, 92, 60)
ramp(pad, 39, 46.9, 50, 62)
ramp(trem, 47, 50.9, 40, 105)
ramp(pad, 51, 57, 72, 96)
ramp(pad, 57, 58.9, 96, 110)
ramp(pad, 59, 72, 100, 112, steps=8)
ramp(pad, 73, 78.9, 100, 55)
CC(pad, 7, 0, 100)
CC(trem, 7, 0, 100)
# cellos: guide-tone counter-line in chorus 1 / final; octave-down melody in the string interlude
GUIDE = {}


def guide_line(inst, ob0, ob1, lo, hi, vel, start):
    prev = start
    for hb in range(int(ob0 * 2), int(ob1 * 2)):
        p0 = hb / 2
        if hb % 2 == 1 and A.C[int(p0)][0] == A.C[int(p0)][1]:
            continue
        span = 1.0 if (hb % 2 == 0 and A.C[int(p0)][0] == A.C[int(p0)][1]) else 0.5
        sym, root, bass, pcs = chord(p0)
        rel = [(p - root) % 12 for p in pcs]
        g = [p for p, r in zip(pcs, rel) if r in (3, 4, 10, 11, 5, 9)] or pcs
        g = avoid(g, p0)
        cands = [m for p in g for m in inst_range(p, lo, hi)]
        if not cands:
            continue
        m = min(cands, key=lambda c: (abs(c - prev), c))
        N(inst, m, p0, p0 + span + 0.02, vel, early=STR_EARLY)
        prev = m


guide_line(vc, 24, 38, 48, 62, 64, 55)
guide_line(vc, 59, 73, 48, 62, 72, 56)
guide_line(vln, 63, 73, 74, 88, 66, 80)     # descant over the second half of the final chorus
for s, m, d in STR_MEL:
    N(vln, m + 12, 51 + s, 51 + s + d + 0.03, 84, early=STR_EARLY)
    N(vc, m - 12, 51 + s, 51 + s + d + 0.03, 70, early=STR_EARLY)
ramp(vln, 51, 57.5, 78, 110)
ramp(vln, 62.9, 64, 50, 95)
ramp(vc, 23.9, 25, 60, 90)
ramp(vc, 50.9, 52, 70, 100)
CC(vln, 7, 0, 100)
CC(vc, 7, 0, 105)
# contrabass
for ob in list(range(20, 38)) + list(range(51, 78)):
    for half in (0, 1):
        p0 = ob + half / 2
        if half == 1 and A.C[ob][0] == A.C[ob][1]:
            continue
        span = 1.0 if (half == 0 and A.C[ob][0] == A.C[ob][1]) else 0.5
        N(cb, bass_of(p0, 28, 43), p0, p0 + span + 0.02, 68 if ob >= 24 else 55, early=STR_EARLY)
CC(cb, 7, 0, 105)
# pizzicato on the off-beats of the bridge
for ob in range(39, 47):
    for k in (1, 3):
        pos = ob + k / 4
        sym, root, bass, pcs = chord(pos)
        for m in voicing(avoid(pcs, pos), 55, 67, None, n=2, center=61):
            N(pizz, m, pos, pos + 0.2, 62, 0.006)
# harp glissando into the key change (Cm7 up two octaves in 32nds)
sym, root, bass, pcs = chord(58.0)
gl = sorted({m for p in pcs for m in inst_range(p, 55, 86)})
for k, m in enumerate(gl):
    pos = 58.0 + k / 32
    if pos < 58.48:
        N(harp, m, pos, pos + 0.4, 55 + k, 0)
# timpani: roll into the final chorus, hits on the big downbeats, final note
tb = nearest(7, 45, 40, 52)    # G (dominant) for the roll
k = 0
pos = 56.5
while pos < 58.48:
    u = (pos - 56.5) / 1.98
    N(timp, tb, pos, pos + 1 / 32, int(30 + 80 * u ** 1.5), 0)
    pos += 1 / 32
for ob in (59, 63, 67, 71):
    N(timp, bass_of(ob, 40, 52), ob, ob + 0.5, 80, 0)
N(timp, bass_of(24, 40, 52), 24, 24.5, 70, 0)
N(timp, nearest(1, 45, 40, 52), 78, 79, 70, 0)       # C# under the last word
for ob, v in [(24, 70), (59, 100), (73, 60)]:
    N(cym, 49, ob, ob + 1, v, 0)

# ---------------------------------------------------------------- render MIDI with FluidSynth
def render(insts, name, gain=0.5):
    mid = pm.PrettyMIDI(resolution=960, initial_tempo=120)
    for ins in insts:
        mid.instruments.append(ins)
    path = f"{OUT}/{name}.mid"
    mid.write(path)
    wav = f"{OUT}/{name}.wav"
    subprocess.run(['fluidsynth', '-ni', '-q', '-R', '0', '-C', '0', '-g', str(gain), '-r', str(SR), '-F', wav, SF2,
                    path], check=True, capture_output=True)
    x, sr = sf.read(wav, dtype='float64')
    assert sr == SR
    return x


N_OUT = IDX(bar_start[NBARS] + 9)


def fit_len(x):
    y = np.zeros((N_OUT, 2))
    n = min(len(x), N_OUT)
    y[:n] = x[:n]
    return y


P = fit_len(render([piano], 'piano'))
S = fit_len(render([pad, trem, vln, vc, cb, pizz], 'strings'))
X = fit_len(render([harp, timp, cym], 'perc'))
# cymbal swell into the key change: a real GM crash, reversed
one = pm.Instrument(program=0, is_drum=True)
one.notes.append(pm.Note(velocity=110, pitch=49, start=0.05, end=4.0))
crash = render([one], 'crash_one')
crash = crash[IDX(0.05):IDX(0.05) + IDX(2 * BAR)][::-1]
crash *= np.linspace(0, 1, len(crash))[:, None] ** 1.5
X[IDX(tm(58.5)) - len(crash):IDX(tm(58.5))] += crash * 0.9
print("rendered: piano notes %d, strings notes %d, perc notes %d" % (
    len(piano.notes), sum(len(i.notes) for i in (pad, trem, vln, vc, cb, pizz)),
    sum(len(i.notes) for i in (harp, timp, cym))))

# ---------------------------------------------------------------- the voice
vsrc, sr = sf.read(f"{STEMS}/04_shutter_doll_vocals.wav", dtype='float64')
env = np.sqrt(np.maximum(ss.oaconvolve((vsrc ** 2).mean(axis=1), np.ones(int(0.02 * SR)) / int(0.02 * SR),
                                        mode='same'), 0))


def quiet_point(sb0, sb1):
    """Lowest-energy instant (20 ms RMS) between source bars sb0 and sb1."""
    a, b = IDX(ST(sb0)), IDX(ST(sb1))
    return (a + int(np.argmin(env[a:b]))) / SR


VOX = np.zeros((N_OUT, 2))
seg_report = []
for a, b, off, semi in A.VOCAL_SEGMENTS:
    t_in = quiet_point(a - 0.06, a + 0.06)
    t_out = quiet_point(b - 0.12, min(b + 0.2, 127.06)) if b < 127 else len(vsrc) / SR
    pad_s = 0.15
    x = vsrc[max(0, IDX(t_in - pad_s)):IDX(t_out + pad_s)].copy()
    if semi:
        x = board(x, [pb.PitchShift(semitones=semi)])
    x = x[IDX(pad_s) if t_in - pad_s >= 0 else 0:]
    x = x[:IDX(t_out - t_in)]
    fi, fo = int(0.03 * SR), int(0.06 * SR)
    x[:fi] *= np.linspace(0, 1, fi)[:, None]
    x[-fo:] *= np.linspace(1, 0, fo)[:, None]
    t_dest = tm(a - off) + (t_in - ST(a))
    i = IDX(t_dest)
    VOX[i:i + len(x)] += x[:N_OUT - i]
    seg_report.append((a, b, round(t_in, 3), round(t_out, 3), round(t_dest, 3), semi))
print("vocal segments (src bar a, b, cut-in s, cut-out s, dest s, semitones):", seg_report)

# downward expander: tame separation bleed between phrases
e = np.sqrt(np.maximum(ss.oaconvolve((VOX ** 2).mean(axis=1), np.ones(int(0.01 * SR)) / int(0.01 * SR),
                                      mode='same'), 0))
thr = db(-44)
g = np.clip(e / thr, 0, 1) ** 1.5
g = ss.oaconvolve(g, np.hanning(int(0.06 * SR)) / np.hanning(int(0.06 * SR)).sum(), mode='same')
VOX *= np.clip(g, 0, 1)[:, None]
VOX = board(VOX, [pb.HighpassFilter(cutoff_frequency_hz=95), pb.HighShelfFilter(cutoff_frequency_hz=8500, gain_db=-1.5),
                  pb.Compressor(threshold_db=-22, ratio=2.5, attack_ms=6, release_ms=120)])


def sec_curve(levels, default=0.0, xf=0.4):
    t = np.arange(N_OUT) / SR
    gcur = np.full(N_OUT, default)
    for name, a, b in A.SECTIONS:
        if name in levels:
            gcur[(t >= tm(a)) & (t < tm(min(b, NBARS)))] = levels[name]
    k = np.hanning(int(xf * SR))
    return ss.oaconvolve(gcur, k / k.sum(), mode='same')


rs = sec_curve({'verse': db(-15), 'chorus': db(-12), 'turn': db(-10), 'bridge': db(-12), 'strings': db(-11),
                'final': db(-11), 'tag': db(-10), 'outro': db(-9)})
VOX_VERB = reverb(VOX * rs[:, None], room=0.74, damp=0.5, predelay=0.03)
throws = np.zeros_like(VOX)
for sb_end, off in [(46.3, 8), (104.0, 53), (127.07, 49)]:
    t1 = tm(sb_end - off)
    a, b = IDX(t1 - BEAT), IDX(t1 + 0.05)
    throws[a:b] = VOX[a:b] * np.hanning(b - a)[:, None]
VOX_THROW = delay(throws, BEAT, feedback=0.45, hp=500, lp=5000)
VOX_ALL = VOX + VOX_VERB + VOX_THROW * db(-6) + reverb(VOX_THROW, 0.8) * db(-10)

# ---------------------------------------------------------------- mix
def rms(x, t0, t1):
    s = x[IDX(t0):IDX(t1)]
    return 20 * np.log10(np.sqrt(np.mean(s ** 2)) + 1e-12)


ch0, ch1 = tm(24), tm(38)
ref = rms(VOX, ch0, ch1)
P = board(P, [pb.HighpassFilter(cutoff_frequency_hz=35), pb.PeakFilter(cutoff_frequency_hz=250, gain_db=-2.0, q=0.8)])
P_ALL = P + reverb(P, room=0.82, damp=0.45, predelay=0.012) * db(-9)
P_ALL *= db(ref - 6.0 - rms(P, ch0, ch1))
S_ALL = S + reverb(S, room=0.88, damp=0.4, predelay=0.02) * db(-4)
S_ALL *= db(ref - 7.0 - rms(S, ch0, ch1))
X_ALL = X + reverb(X, room=0.88, damp=0.4) * db(-4)
X_ALL *= db(ref - 8.0 - rms(X, tm(56), tm(60)))
P_ALL *= sec_curve({'verse': db(4), 'chorus': db(-1), 'final': db(-3), 'bridge': db(2)}, default=1.0)[:, None]
S_ALL *= sec_curve({'chorus': db(-1), 'final': db(-2.5), 'bridge': db(1)}, default=1.0)[:, None]
mix = VOX_ALL + P_ALL + S_ALL + X_ALL
mix = compress(mix, thresh=-18, ratio=1.8, attack=25, release=250)
end_t = tm(82) + 7.5
mix = mix[:IDX(end_t)]
k = int(2.0 * SR)
mix[-k:] *= np.linspace(1, 0, k)[:, None] ** 2


def lufs(x):
    tmp = f"{OUT}/_tmp.wav"
    sf.write(tmp, x.astype(np.float32), SR)
    r = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', tmp, '-af', 'ebur128', '-f', 'null', '-'],
                       capture_output=True, text=True)
    return float([ln for ln in r.stderr.splitlines() if ln.strip().startswith('I:')][-1].split()[1])


TARGET = -11.5
gdb = TARGET - lufs(mix)
for _ in range(3):
    out, gr = limiter(mix * db(gdb), ceiling_db=-1.0)
    L = lufs(out)
    if abs(L - TARGET) < 0.15:
        break
    gdb += TARGET - L
mix = out
print(f"LUFS {lufs(mix):.2f}, true peak {true_peak_db(mix):.2f} dBTP, max GR {20 * np.log10(gr.min()):.2f} dB, "
      f"length {len(mix) / SR:.1f} s")
for name, a, b in A.SECTIONS:
    print(f"  {name:8s} {tm(a):6.1f}-{tm(min(b, NBARS)):6.1f}s  voice {rms(VOX_ALL, tm(a), tm(min(b, NBARS))):6.1f}  "
          f"piano {rms(P_ALL, tm(a), tm(min(b, NBARS))):6.1f}  strings {rms(S_ALL, tm(a), tm(min(b, NBARS))):6.1f}  "
          f"mix {rms(mix, tm(a), tm(min(b, NBARS))):6.1f}")
sf.write(f"{OUT}/Shutter_Doll_Afterimage.wav", mix.astype(np.float32), SR, subtype='PCM_24')
json.dump({name: [tm(a), tm(min(b, NBARS))] for name, a, b in A.SECTIONS}, open(f"{OUT}/_sections.json", 'w'))
np.savez_compressed(f"{OUT}/_stems.npz", vox=VOX_ALL[:len(mix)].mean(1)[::4].astype(np.float32),
                    piano=P_ALL[:len(mix)].mean(1)[::4].astype(np.float32),
                    strings=S_ALL[:len(mix)].mean(1)[::4].astype(np.float32))
json.dump({'parts': {ins.name: [(n.start, n.pitch) for n in ins.notes] for ins in
                     (piano, pad, trem, vln, vc, cb, pizz, harp, timp)},
           'voice': [(tm(nt['slot'] / 16 - off) , nt['midi'] + semi) for a, b, off, semi in A.VOCAL_SEGMENTS
                     for nt in mel if a <= nt['slot'] / 16 < b]}, open(f"{OUT}/_notes.json", 'w'))
print("done")
