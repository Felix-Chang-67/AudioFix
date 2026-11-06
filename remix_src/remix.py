"""Shutter Doll (Clockwork Darkroom Remix) — arrangement & mix.

usage: python3 -I remix.py <scripts_dir> <stems_dir> <out_dir>

Source grid (measured): 145.01 BPM, source bar 0 starts at 0.015 s, C minor, tuned ~+6 cents.
Output bar ob == source bar sb + 2 for sb in [8, 127]; ob 0-9 is a new intro, ob 129+ a new ending.
"""
import sys
import json
import subprocess
import numpy as np
import soundfile as sf
import librosa
import scipy.signal as ss

sys.path.insert(0, sys.argv[1])
from remixlib.dsp import (SR, db, to_stereo, pan, add_at, svf, butter, reverb, delay, compress, board,
                          tape_stop, automation, exp_automation, true_peak_db, limiter)
from remixlib import instruments as I
import pedalboard as pb

STEMS_DIR, OUT_DIR = sys.argv[2], sys.argv[3]

# ------------------------------------------------------------------ grid
BPM = 145.01
BEAT = 60 / BPM
BAR = 4 * BEAT
S16 = BEAT / 4
SRC_PH = 0.015
OUT_PH = 0.20
OFF = 2                       # ob = sb + OFF
SRC_FIRST, SRC_LAST = 8, 127  # source bars used (127 = file end, ~0.06 bar long)
END_OB = 139


def T(ob):
    return OUT_PH + ob * BAR


def ST(sb):
    return SRC_PH + sb * BAR


def IDX(t):
    return int(round(t * SR))


N = IDX(T(END_OB) + 7.0)
LEAD = 0.015  # stem switches happen 15 ms before the bar line (source timing jitter is ~±10 ms)

# ------------------------------------------------------------------ load stems
NAMES = ['bass', 'drums', 'other', 'vocals', 'guitar', 'piano']
SRC = {}
for i, n in enumerate(NAMES):
    x, sr = sf.read(f"{STEMS_DIR}/0{i + 1}_shutter_doll_{n}.wav", dtype='float64')
    assert sr == SR
    SRC[n] = x
SHIFT = IDX(T(SRC_FIRST + OFF)) - IDX(ST(SRC_FIRST))
print(f"shift = {SHIFT} samples ({SHIFT / SR:.4f} s)")

STEM = {}
for n, x in SRC.items():
    y = np.zeros((N, 2))
    a = IDX(ST(SRC_FIRST) - LEAD)
    y[a + SHIFT:a + SHIFT + len(x) - a] = x[a:]
    STEM[n] = y


def seg_gain(segs, default=0.0, xf=0.006, lead=LEAD):
    """Step gain curve over output bars with short raised-cosine crossfades."""
    g = np.full(N, float(default))
    for ob0, ob1, val in segs:
        g[max(0, IDX(T(ob0) - lead)):IDX(T(ob1) - lead)] = val
    k = np.hanning(max(3, int(xf * SR)))
    return ss.oaconvolve(g, k / k.sum(), mode='same')


def ob_auto(points, log=False):
    f = exp_automation if log else automation
    return f(points, N, t0=OUT_PH, unit=BAR)


def src_slice(name, t0, t1):
    return SRC[name][IDX(t0):IDX(t1)].copy()


def fade(x, fi=0.003, fo=0.01):
    x = to_stereo(x).copy()
    a, b = int(fi * SR), int(fo * SR)
    if a > 0:
        x[:a] *= np.linspace(0, 1, a)[:, None]
    if b > 0:
        x[-b:] *= np.linspace(1, 0, b)[:, None]
    return x


def rms_db(x, t0, t1):
    s = x[IDX(t0):IDX(t1)]
    return 20 * np.log10(np.sqrt(np.mean(s ** 2)) + 1e-12)


def bus():
    return np.zeros((N, 2))


# ------------------------------------------------------------------ sections (output bars)
SEC = dict(wind=(0, 2), intro=(2, 10), riff=(10, 18), verse1=(18, 34), ch1=(34, 48), inter=(48, 56),
           v2a=(56, 64), v2b=(64, 72), ch2=(72, 86), brk=(86, 94), bridge=(94, 102), build=(102, 110),
           drop=(110, 112), final=(112, 124), tag=(124, 129), outro=(129, END_OB))

# ------------------------------------------------------------------ original stem arrangement
INST = ['drums', 'bass', 'guitar', 'other', 'piano']
full = [(10, 129, 1.0)]
drum_plan = [(10, 18, 1), (18, 34, 0), (34, 56, 1), (56, 64, 0), (64, 70, 1), (70, 72, 0), (72, 87, 1),
             (87, 94, 0), (94, 102, 0), (102, 106, 1), (106, 110, 0), (110, 112, 0), (112, 129, 1)]
band_plan = [(10, 87, 1), (87, 94, 0), (94, 110, 1), (110, 112, 0), (112, 129, 1)]
G = {
    'drums': seg_gain(drum_plan),
    'bass': seg_gain(band_plan),
    'guitar': seg_gain(band_plan),
    'other': seg_gain(band_plan),
    'piano': seg_gain(band_plan),
    # vocals: muted before verse 1 (only separation bleed there); allowed to run to the file end
    'vocals': seg_gain([(18, 110, 1.0), (110, 112, db(-2.5)), (112, 129.03, 1.0)], xf=0.03),
}
# the last bar before verse 1 / v2a / bridge keeps the guitar a touch quieter in halftime parts
G['guitar'] *= seg_gain([(18, 34, 0.82), (94, 102, 0.85)], default=1.0, xf=0.05)

inst = bus()
for n in INST:
    inst += STEM[n] * G[n][:, None]
vox = STEM['vocals'] * G['vocals'][:, None]

# interlude: low-pass the band, opening up over 8 bars
lp = ob_auto([(0, 20000), (47.95, 20000), (48, 420), (52, 1800), (55.5, 9000), (55.95, 20000), (END_OB, 20000)], log=True)
seg = slice(IDX(T(47.9)), IDX(T(56.05)))
inst[seg] = svf(inst[seg], lp[seg], 0.9, 'lp')

# build-ups: high-pass sweeps on the band (two cascaded 12 dB/oct stages)
hp = ob_auto([(0, 15), (32, 15), (33.97, 380), (34, 15), (70, 15), (71.97, 380), (72, 15),
              (104, 15), (109.97, 900), (110, 15), (END_OB, 15)], log=True)
for a, b in [(31.9, 34.05), (69.9, 72.05), (103.9, 110.05)]:
    seg = slice(IDX(T(a)), IDX(T(b)))
    inst[seg] = svf(svf(inst[seg], hp[seg], 0.707, 'hp'), hp[seg], 0.707, 'hp')

# end of chorus 2: let the downbeat land, then tape-stop the band over 2 beats
TS0 = T(86) + BEAT
seg = slice(0, IDX(T(87.5)))
inst[seg] = tape_stop(inst[seg], TS0, 2 * BEAT, curve=1.6)

# ------------------------------------------------------------------ reference levels
REF_MIX = rms_db(inst + vox, T(34), T(48))       # chorus-1 level of the original band
REF_DRUM_PK = np.percentile(np.abs(STEM['drums'][IDX(T(34)):IDX(T(48))]), 99.97)
print(f"chorus1 stem mix rms {REF_MIX:.1f} dB, drum peak ref {REF_DRUM_PK:.3f}")

# ------------------------------------------------------------------ programmed drums
kick_b, snare_b, hat_b, perc_b = bus(), bus(), bus(), bus()
K, KI, SN, HC, HO = I.kick(), I.industrial_kick(), I.big_snare(), I.hat(), I.hat(open_=True)
SH_TICK, SH_FULL = I.shutter(False), I.shutter(True)
CLAP = I.clap()


def step_t(ob, step):
    return T(ob) + step * S16


def halftime_bar(ob, var=0, level=1.0, hats=True):
    kicks = {0: [0, 10], 1: [0, 3, 10], 2: [0, 10, 13], 3: [0, 7, 10]}[var]
    for s in kicks:
        add_at(kick_b, K, step_t(ob, s), level * (1.0 if s == 0 else 0.8))
    add_at(snare_b, SN, step_t(ob, 8), level)
    if hats:
        for s in range(0, 16, 2):
            v = 0.9 if s % 4 == 2 else 0.55
            add_at(hat_b, pan(HC, 0.25), step_t(ob, s), level * v)
            add_at(perc_b, pan(SH_TICK, -0.35), step_t(ob, s + 1), level * 0.35 * (1 if s % 4 == 2 else 0.6))
        if var in (1, 3):  # 16th / 32nd hat roll into the next bar
            for k in range(6):
                add_at(hat_b, pan(HC, 0.25), step_t(ob, 13 + k * 0.5), level * (0.35 + 0.08 * k))
        add_at(perc_b, pan(SH_FULL, 0.0), step_t(ob, 12), level * 0.55)


def industrial_bar(ob, level=1.0):
    for s in (0, 4, 8, 12):
        add_at(kick_b, KI, step_t(ob, s), level * 0.9)
    for s in (4, 12):
        add_at(snare_b, CLAP, step_t(ob, s), level * 0.8)
    for s in range(16):
        if s % 4 == 2:
            add_at(hat_b, pan(HO, 0.3), step_t(ob, s), level * 0.45)
        else:
            add_at(perc_b, pan(SH_TICK, -0.3 if s % 2 else 0.3), step_t(ob, s), level * (0.32 if s % 2 else 0.2))


def snare_roll(ob0, ob1, rates, level0=0.25, level1=1.0, pitch_up=True):
    """rates: list of (ob_start, subdivision per bar). Crescendo + rising pitch."""
    t, end = T(ob0), T(ob1)
    while t < end - 1e-6:
        ob = (t - OUT_PH) / BAR
        div = [r for (o, r) in rates if o <= ob + 1e-9][-1]
        u = (t - T(ob0)) / (end - T(ob0))
        lvl = level0 + (level1 - level0) * u ** 1.5
        s = I.snare(dur=0.25, tone=180 * (2 ** (u * 1.0) if pitch_up else 1))
        add_at(snare_b, pan(s, 0.0), t, lvl)
        t += BAR / div


def shutter_offbeats(ob0, ob1, level=0.3):
    for ob in range(ob0, ob1):
        for s in (2, 6, 10, 14):
            add_at(perc_b, pan(SH_TICK, 0.4 if s in (2, 10) else -0.4), step_t(ob, s), level)


# verse 1 (halftime) + its build
for i, ob in enumerate(range(18, 32)):
    halftime_bar(ob, var=[0, 1, 2, 3][i % 4])
halftime_bar(32, var=0, hats=False)
snare_roll(32.5, 34, [(32.5, 8), (33, 16), (33.5, 32)], 0.2, 0.9)
# verse 2a: industrial; 2b: shutter offbeats over the band; build
for ob in range(56, 64):
    industrial_bar(ob)
shutter_offbeats(64, 70, 0.28)
for ob in (70, 71):
    for s in (0, 4, 8, 12):
        add_at(kick_b, K, step_t(ob, s), 0.85)
snare_roll(70, 72, [(70, 8), (71, 16), (71.5, 32)], 0.2, 0.95)
# break: beat creeps back in
for ob in (90, 91):
    add_at(kick_b, K, step_t(ob, 0), 0.7)
    for s in range(0, 16, 2):
        add_at(hat_b, pan(HC, 0.25), step_t(ob, s), 0.35 + 0.02 * s)
halftime_bar(92, 0, 0.9)
halftime_bar(93, 3, 0.95)
# bridge
for i, ob in enumerate(range(94, 102)):
    halftime_bar(ob, var=[0, 1, 2, 3][i % 4])
# main build: roll over the band, then four-on-the-floor + roll without the original drums
snare_roll(102, 106, [(102, 4), (104, 8)], 0.15, 0.45)
for ob in range(106, 110):
    for s in (0, 4, 8, 12):
        add_at(kick_b, K, step_t(ob, s), 0.75 + 0.05 * (ob - 106))
snare_roll(106, 110, [(106, 8), (107, 16), (109, 32)], 0.45, 1.0)
# final chorus sparkle
shutter_offbeats(112, 128, 0.22)
for s_ in (2, 6, 10):
    add_at(perc_b, pan(SH_TICK, 0.4 if s_ in (2, 10) else -0.4), step_t(128, s_), 0.22)

# ------------------------------------------------------------------ chord detection for pads
PC = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
CHORDS = {
    'Ab': (['Ab3', 'C4', 'Eb4', 'Ab4'], [8, 0, 3]),
    'Bb': (['Bb3', 'D4', 'F4', 'Bb4'], [10, 2, 5]),
    'Cm': (['C4', 'Eb4', 'G4', 'C5'], [0, 3, 7]),
    'Eb': (['Bb3', 'Eb4', 'G4', 'Bb4'], [3, 7, 10]),
    'Fm': (['F3', 'Ab3', 'C4', 'F4'], [5, 8, 0]),
    'Gm': (['G3', 'Bb3', 'D4', 'G4'], [7, 10, 2]),
    'C5': (['C4', 'G4', 'C5'], [0, 7]),
}
# The three choruses are the same 14-bar progression (verified by the detector below, which only
# disagrees on two low-confidence bars). Final chorus differs in its first bar (Cm instead of Ab).
CHORUS = ['Ab', 'Bb', 'Cm', 'Cm', 'Ab', 'Bb', 'Cm', 'Cm', 'Ab', 'Bb', 'Bb', 'Cm', 'Ab', 'Bb']
harm = (SRC['guitar'] + SRC['other'] + 0.5 * SRC['bass']).mean(axis=1)
hy = librosa.resample(harm, orig_sr=SR, target_sr=11025)
C = librosa.feature.chroma_cqt(y=hy, sr=11025, hop_length=256, fmin=librosa.note_to_hz('C2'), n_octaves=5)
ct = librosa.times_like(C, sr=11025, hop_length=256)


def detect_chord(sb):
    sel = (ct >= ST(sb) + 0.05) & (ct < ST(sb + 1) - 0.05)
    c = C[:, sel].mean(1)
    c = c / (np.linalg.norm(c) + 1e-9)
    best, score = None, -1
    for name, (_, pcs) in CHORDS.items():
        v = np.zeros(12)
        v[pcs] = 1
        s = float(c @ (v / np.linalg.norm(v)))
        if s > score:
            best, score = name, s
    return best, score


chart = {}
for a, b in [(32, 46), (70, 84), (108, 127)]:
    for sb in range(a, b):
        chart[sb] = detect_chord(sb)
print("pad chord chart (source bar: chord score):")
print("  " + "  ".join(f"{sb}:{c}{s:.2f}" for sb, (c, s) in chart.items()))

# ------------------------------------------------------------------ pads
pad_b = bus()


def render_pad_run(chords, ob0, level=1.0, cutoff=2600.0, attack=0.03, release=0.3):
    """chords: list of chord names per bar starting at ob0 (None = rest). Merges repeats."""
    i = 0
    while i < len(chords):
        c = chords[i]
        j = i
        while j + 1 < len(chords) and chords[j + 1] == c:
            j += 1
        if c is not None:
            notes = [I.nm(x) for x in CHORDS[c][0]]
            p = I.pad_chord(notes, (j - i + 1) * BAR - 0.02, attack=attack, release=release, cutoff=cutoff)
            add_at(pad_b, p, T(ob0 + i), level)
        i = j + 1


agree = sum(chart[a + i][0] == c for a in (32, 70) for i, c in enumerate(CHORUS)) + \
    sum(chart[110 + i][0] == c for i, c in enumerate(CHORUS[2:]))
print(f"detector agrees with canonical chorus chart on {agree}/40 bars")
render_pad_run(CHORUS, 32 + OFF)
render_pad_run(CHORUS, 70 + OFF)
render_pad_run(CHORUS[2:], 110 + OFF, level=0.85)
add_at(pad_b, I.pad_chord([I.nm(x) for x in CHORDS['C5'][0]], 4.875 * BAR - 0.03, release=0.12), T(124), 0.85)
HOOK_CHORDS = ['Ab', 'Bb', 'Cm', 'Cm']
render_pad_run(HOOK_CHORDS * 2, 2, level=0.35, cutoff=1100, attack=0.35, release=0.8)   # intro
render_pad_run(['Cm'] + HOOK_CHORDS + ['Ab', 'Bb'], 87, level=0.4, cutoff=1300, attack=0.4, release=0.8)  # break
# sidechain pump on quarter notes
t = np.arange(N) / SR
ph = ((t - OUT_PH) % BEAT)
pump = 1 - 0.55 * np.exp(-ph / 0.085) * (1 - np.exp(-ph / 0.004))
pump_mask = seg_gain([(34, 48, 1), (72, 86, 1), (112, 129, 1)], xf=0.02)
pad_b *= (1 - pump_mask * (1 - pump))[:, None]
# the drop: soft held chords under the a-cappella vocal (no pump)
render_pad_run([chart[108][0], chart[109][0]], 110, level=0.4, cutoff=1500, attack=0.25, release=0.6)

# ------------------------------------------------------------------ music box
mbox_b = bus()
HOOK = [(0, 'G5'), (4, 'Eb6'), (12, 'Eb6'),
        (16, 'Eb6'), (18, 'D6'), (24, 'D6'), (28, 'Ab5'), (30, 'G5'),
        (32, 'G5'), (36, 'Ab5'), (38, 'Bb5'), (40, 'Ab5'), (41, 'G5'), (43, 'Ab5'), (45, 'Eb5'), (47, 'D5'),
        (48, 'Eb5'), (50, 'D5'), (52, 'Eb5'), (56, 'G5'), (58, 'C5')]
ARP = {'Ab': ['Ab3', 'Eb4', 'Ab4', 'Eb4', 'C5', 'Eb4', 'Ab4', 'Eb4'],
       'Bb': ['Bb3', 'F4', 'Bb4', 'F4', 'D5', 'F4', 'Bb4', 'F4'],
       'Cm': ['C4', 'G4', 'C5', 'G4', 'Eb5', 'G4', 'C5', 'G4']}
NOTE_CACHE = {}


def mb(note, vel, t, p=0.0):
    key = (note, round(vel, 3))
    if key not in NOTE_CACHE:
        NOTE_CACHE[key] = I.music_box_note(I.nm(note), vel)
    add_at(mbox_b, pan(NOTE_CACHE[key], p), t)


def mbox_phrase(ob0, melody=True, arps=True, timefn=None, arp_every=1, mel_vel=1.0, arp_vel=0.42):
    tf = timefn or (lambda p: T(ob0) + p * S16)
    if melody:
        for p, n in HOOK:
            mb(n, mel_vel, tf(p), -0.12)
    if arps:
        for bar, ch in enumerate(HOOK_CHORDS):
            for k, n in enumerate(ARP[ch]):
                if k % arp_every == 0:
                    mb(n, arp_vel * (1.15 if k == 0 else 1.0), tf(bar * 16 + k * 2), 0.28)


mbox_phrase(2)                       # intro: hook
mbox_phrase(6, melody=False)         # intro: arps under the radio vocal
def mbox_arps(ob, chord, vel=0.42):
    for k, n in enumerate(ARP[chord]):
        mb(n, vel * (1.15 if k == 0 else 1.0), T(ob) + k * 2 * S16, 0.28)


mbox_arps(87, 'Cm', 0.36)            # break: lead-in after the tape stop
mbox_phrase(88)                      # break: hook
mbox_arps(92, 'Ab')
mbox_arps(93, 'Bb')
# outro: the spring runs down — ritardando, thinning arps
O0 = T(131)
A_RIT = 2.4


def rit(p):
    return O0 + S16 * (p + A_RIT * p ** 3 / (3 * 64 ** 2))


mbox_phrase(131, timefn=rit, arp_every=2, mel_vel=0.9, arp_vel=0.38)
mb('C4', 0.55, rit(58), 0.28)
OUTRO_END = rit(58) + 3.2

# ------------------------------------------------------------------ vocal-derived parts
fx_b, chop_b, radio_b = bus(), bus(), bus()
# radio teaser in the intro: chorus-1 vocal (source bars 32-35) through an old radio
rv = src_slice('vocals', ST(32), ST(36)).mean(axis=1)
rv = butter(rv, [480, 3200], 'band', 3)
rv = np.tanh(3.0 * rv / (np.abs(rv).max() + 1e-9)) * 0.5
rv = fade(rv, fi=BEAT, fo=0.25)
add_at(radio_b, rv, T(6))

# vocal chops for the interlude, cut from held notes (checked with pYIN in analysis)
C5_SRC = ST(44) + 6 * S16      # held C5
G4_SRC = ST(34) + 0 * S16      # held G4


def chop(t_src, length, semis):
    """Slice a held vocal note and retune it to the band (the singer is ~20-25 cents sharp here)."""
    pad = 0.05
    x = src_slice('vocals', t_src + 0.01 - pad, t_src + 0.01 + length + pad)
    x = board(x, [pb.PitchShift(semitones=semis)])
    return fade(x[int(pad * SR):int(pad * SR) + int(length * SR)], 0.003, 0.018)


CH_C16, CH_C8 = chop(C5_SRC, S16 * 0.95, -0.24), chop(C5_SRC, S16 * 1.9, -0.24)
CH_G16, CH_G8 = chop(G4_SRC, S16 * 0.95, -0.14), chop(G4_SRC, S16 * 1.9, -0.14)
PAT = [{0: 'c', 3: 'C', 4: 'C', 6: 'g', 8: 'c', 11: 'C', 12: 'C', 14: 'G'},
       {0: 'c', 3: 'C', 6: 'C', 8: 'g', 10: 'C', 11: 'C', 12: 'c', 14: 'C', 15: 'C'}]
SMP = {'C': CH_C16, 'c': CH_C8, 'G': CH_G16, 'g': CH_G8}
for i, ob in enumerate(range(48, 56)):
    for s, k in PAT[i % 2].items():
        add_at(chop_b, SMP[k], step_t(ob, s), 1.0 if s == 0 else 0.85)

# vocal sends (delay + reverb), automated per section
dsend = seg_gain([(18, 34, db(-15)), (94, 102, db(-14)), (110, 112, db(-10))], xf=0.05)
rsend = seg_gain([(18, 34, db(-12)), (34, 48, db(-16)), (72, 86, db(-16)), (94, 102, db(-12)),
                  (110, 112, db(-6)), (112, 129.2, db(-15))], xf=0.05)
vox_delay = delay(vox * dsend[:, None], 0.75 * BEAT, feedback=0.38, hp=350, lp=4500)
# echo throws on the last word of chorus 2 and on the very last word of the song
throws = bus()
for a, b in [(T(86), T(86) + 2 * BEAT), (T(128) + 2 * BEAT, T(129) + 0.06)]:
    throws[IDX(a):IDX(b)] = fade(vox[IDX(a):IDX(b)], 0.005, 0.03)
vox_throw = delay(throws, 0.75 * BEAT, feedback=0.55, hp=400, lp=4000)
vox_verb = reverb(vox * rsend[:, None] + 0.5 * vox_throw, room=0.86, damp=0.45)

# ------------------------------------------------------------------ FX
for ob in (10, 34, 72, 112):
    add_at(fx_b, I.impact(), T(ob), 0.33)
    add_at(fx_b, I.crash(), T(ob), 0.3)
    add_at(fx_b, I.reverse_crash(BAR), T(ob - 1), 0.3)
for a, b, lvl in [(8, 10, 0.5), (32, 34, 0.45), (70, 72, 0.45), (104, 110, 0.55), (111, 112, 0.3)]:
    add_at(fx_b, I.riser(T(b) - T(a)), T(a), lvl)
# intro: winding key, clock ticks, shutter on beat 4
for k, ob in enumerate([0.0, 0.5, 1.05]):
    add_at(perc_b, pan(I.ratchet(n_clicks=[8, 8, 5][k]), 0.15), T(ob), 1.6)
for ob in range(2, 10):
    for b4 in range(4):
        add_at(perc_b, pan(I.tick(hi=b4 % 2 == 0), 0.45 if b4 % 2 == 0 else -0.45), T(ob) + b4 * BEAT, 0.3)
    if ob >= 6:
        add_at(perc_b, SH_FULL, T(ob) + 3 * BEAT, 0.4)
# the drop: shutter + flash, everything else stops
add_at(fx_b, SH_FULL, T(110), 1.0)
add_at(fx_b, I.flash(), T(110) + 0.065, 0.6)
add_at(fx_b, I.film_advance(), T(110) + 0.3, 0.35)
# crackle beds
crk = bus()
for a, b, lvl in [(0, 10, 0.5), (86.3, 94, 0.4), (129.3, None, 0.45)]:
    t1 = T(b) if b else OUTRO_END + 3.0
    c = I.crackle(t1 - T(a), level=lvl)
    ramp = np.minimum(1, np.minimum(np.arange(len(c)) / (0.8 * SR), (len(c) - np.arange(len(c))) / (1.5 * SR)))
    add_at(crk, c * ramp[:, None], T(a))

# ------------------------------------------------------------------ the ending
# final hit = the band's own C-minor downbeat (source bar 110, final chorus) + impact, pad, sub
hit = sum(src_slice(n, ST(110) - 0.01, ST(110) + 1.6) for n in INST)
env = np.exp(-np.arange(len(hit)) / SR / 0.32)
hit = hit * env[:, None]
hit[:int(0.003 * SR)] *= np.linspace(0, 1, int(0.003 * SR))[:, None]
end_b = bus()
add_at(end_b, hit, T(129) - 0.01)
add_at(end_b, I.impact(3.0), T(129), 0.9)
add_at(end_b, I.crash(4.0), T(129), 0.6)
add_at(end_b, I.pad_chord([I.nm(x) for x in ['C3', 'G3', 'C4', 'Eb4', 'G4', 'C5']], 0.6, attack=0.01,
                          release=2.5, cutoff=2200), T(129), 1.0)
add_at(end_b, I.sub_note(I.nm('C2'), 0.4, release=1.5), T(129), 0.5)
# the band itself stops right before the final downbeat (vocals ring into the hit)
cut = seg_gain([(0, 128.875, 1.0)], xf=0.008)
inst *= cut[:, None]
# last sound: the camera
add_at(fx_b, SH_FULL, OUTRO_END + 0.5, 0.6)
add_at(fx_b, I.film_advance(0.6), OUTRO_END + 0.85, 0.35)

# ------------------------------------------------------------------ bus processing
def level_to(x, target_db, t0, t1):
    cur = rms_db(x, t0, t1)
    return x * db(target_db - cur), cur


drums_prog = kick_b * 1.0 + snare_b * 0.8 + hat_b * 0.22 + perc_b * 0.5
snare_verb = reverb(snare_b * 0.8 + perc_b * 0.3, room=0.6, damp=0.55)
drums_prog = drums_prog + snare_verb * db(-5)
drums_prog = compress(drums_prog, thresh=-14, ratio=3, attack=8, release=120)
# match the programmed beat to the original drum level (verse-1 halftime vs chorus-1 original)
drums_prog, _ = level_to(drums_prog, rms_db(STEM['drums'], T(34), T(48)) - 2.0, T(18), T(32))

pad_wet = reverb(pad_b, room=0.8, damp=0.5) * db(-4)
pad_all, _ = level_to(pad_b + pad_wet, REF_MIX - 15.0, T(34), T(48))

mbox_all = mbox_b + reverb(mbox_b, room=0.88, damp=0.35) * db(0)
mbox_all = board(mbox_all, [pb.Chorus(rate_hz=0.6, depth=0.12, centre_delay_ms=8, feedback=0.0, mix=0.3)])
mbox_all, _ = level_to(mbox_all, REF_MIX - 12.0, T(2), T(6))

radio = radio_b + reverb(radio_b, room=0.7, damp=0.6) * db(-5)
radio, _ = level_to(radio, REF_MIX - 15.5, T(6.5), T(10))

chops = chop_b + delay(chop_b, 0.5 * BEAT, 0.3, 400, 6000) * db(-8) + reverb(chop_b, 0.7) * db(-6)
chops = butter(chops, 220, 'high', 2)
chops, _ = level_to(chops, REF_MIX - 11.0, T(48), T(56))

fx_all = fx_b + reverb(fx_b, room=0.85, damp=0.5) * db(-4)
fx_all *= db(REF_MIX + 9.0)  # fx are peak-normalised one-shots; scale relative to the mix
crk *= db(REF_MIX - 12.0)
end_all = end_b + reverb(end_b, room=0.86, damp=0.5) * db(-1)
end_all, _ = level_to(end_all, REF_MIX + 2.5, T(129), T(129) + 0.6)

vox_all = vox + vox_delay + vox_verb

mix = inst + vox_all + drums_prog + pad_all + mbox_all + radio + chops + fx_all + crk + end_all

# ------------------------------------------------------------------ master
mix = compress(mix, thresh=-16, ratio=2.0, attack=20, release=250)
mix = mix[:IDX(OUTRO_END + 4.5)]
tail = int(1.0 * SR)
mix[-tail:] *= np.linspace(1, 0, tail)[:, None] ** 2

# loudness match against the original (sum of stems), measured with ffmpeg ebur128
def lufs(x):
    tmp = f"{OUT_DIR}/_tmp_lufs.wav"
    sf.write(tmp, x.astype(np.float32), SR)
    r = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', tmp, '-af', 'ebur128=peak=true', '-f', 'null', '-'],
                       capture_output=True, text=True)
    line = [ln for ln in r.stderr.splitlines() if ln.strip().startswith('I:')][-1]
    return float(line.split()[1])


orig = sum(SRC.values())
L_orig_int = lufs(orig)
L_orig_ch = lufs(orig[IDX(ST(32)):IDX(ST(46))])
target_ch = L_orig_ch - 1.0
g_db = target_ch - lufs(mix[IDX(T(34)):IDX(T(48))])
for it in range(4):
    out_mix, gr = limiter(mix * db(g_db), ceiling_db=-1.0)
    L_ch = lufs(out_mix[IDX(T(34)):IDX(T(48))])
    print(f"  iter {it}: gain {g_db:+.2f} dB -> chorus LUFS {L_ch:.2f} (target {target_ch:.2f}), "
          f"max GR {20 * np.log10(gr.min()):.1f} dB")
    if abs(L_ch - target_ch) < 0.15:
        break
    g_db += target_ch - L_ch
mix = out_mix
print(f"original: integrated {L_orig_int:.2f} LUFS, chorus-1 {L_orig_ch:.2f} LUFS")
print(f"remix:    integrated {lufs(mix):.2f} LUFS, true peak {true_peak_db(mix):.2f} dBTP, length {len(mix) / SR:.1f} s")
gr_db = 20 * np.log10(gr)
for name, (a0, b0) in SEC.items():
    sl = gr_db[IDX(T(a0)):IDX(T(b0))]
    if len(sl):
        print(f"  limiter GR {name:7s}: mean {sl.mean():5.2f} dB, max {sl.min():6.2f} dB")

out = f"{OUT_DIR}/Shutter_Doll_Clockwork_Darkroom_Remix.wav"
sf.write(out, mix.astype(np.float32), SR, subtype='PCM_24')
np.save(f"{OUT_DIR}/_marks.npy", np.array([T(k) for k in range(END_OB + 1)]))
json.dump({k: [T(v[0]), T(v[1])] for k, v in SEC.items()}, open(f"{OUT_DIR}/_sections.json", 'w'), indent=1)
# stem-level debug renders for checking
np.savez_compressed(f"{OUT_DIR}/_buses.npz", **{k: v[:len(mix)].mean(axis=1).astype(np.float32)[::4] for k, v in dict(
    inst=inst, vox=vox_all, prog=drums_prog, pad=pad_all, mbox=mbox_all, radio=radio, chops=chops, fx=fx_all,
    end=end_all).items()})
print("wrote", out)
