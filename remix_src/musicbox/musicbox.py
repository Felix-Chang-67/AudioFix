"""Shutter Doll — Music Box Version.

usage: python3 -I musicbox.py <scripts_dir> <melody.json> <chart.json> <out_dir>

Everything is synthesised: the vocal melody (transcribed on the source 16th grid), a harmony voice,
bass and arpeggios derived from the measured chord chart, all played on a modelled comb music box.
"""
import sys
import json
import subprocess
import numpy as np
import soundfile as sf
import scipy.signal as ss

sys.path.insert(0, sys.argv[1])
from remixlib.dsp import SR, RNG, db, pan, add_at, butter, noise, reverb, compress, limiter, true_peak_db, svf
from remixlib import instruments as I

MEL_FILE, CHART_FILE, OUT_DIR = sys.argv[2], sys.argv[3], sys.argv[4]
SPEED = 0.8                                   # 145.01 BPM -> 116.0 BPM
BPM = 145.01 * SPEED
S16 = 15 / BPM
BAR = 16 * S16
PRE = 2.6                                     # seconds of wind-up before the music starts

# ---------------------------------------------------------------- form (source bars)
BLOCKS = [  # (name, src_bar_start, src_bar_end_exclusive)
    ('intro', None, 4), ('verse', 16, 32), ('chorus', 32, 46), ('inter', 46, 50), ('verse', 54, 70),
    ('chorus', 70, 85), ('bridge', 92, 104), ('build', 104, 108), ('final', 108, 122), ('tag', 122, 127),
]
out_bar = 0
LAYOUT = []
for name, a, b in BLOCKS:
    n = b if a is None else b - a
    LAYOUT.append((name, a, b, out_bar))
    out_bar += n
TOTAL_BARS = out_bar
RIT_BARS = 2                                  # the spring runs down over the last two bars
RIT_A = 1.6


def slot_time(oslot):
    """Output 16th index -> seconds, with the closing ritardando."""
    s0 = (TOTAL_BARS - RIT_BARS) * 16
    if oslot <= s0:
        return PRE + oslot * S16
    u = (oslot - s0) / (RIT_BARS * 16)
    L = RIT_BARS * 16
    return PRE + s0 * S16 + S16 * L * (u + RIT_A * u ** 3 / 3)


# ---------------------------------------------------------------- harmony
PCS = {'Cm': [0, 3, 7], 'Eb': [3, 7, 10], 'Fm': [5, 8, 0], 'F': [5, 9, 0], 'Gm': [7, 10, 2], 'G': [7, 11, 2],
       'Ab': [8, 0, 3], 'Bb': [10, 2, 5], 'C5': [0, 7], 'F5': [5, 0], 'G5': [7, 2], 'Ab5': [8, 3], 'Bb5': [10, 5],
       'Eb5': [3, 10], 'D5': [2, 9]}
CHORUS = ['Ab', 'Bb', 'Cm', 'Cm', 'Ab', 'Bb', 'Cm', 'Cm', 'Ab', 'Bb', 'Bb', 'Cm', 'Ab', 'Bb']
chart = json.load(open(CHART_FILE))
FIXED = {}
for a in (32, 70):
    for i, c in enumerate(CHORUS):
        FIXED[a + i] = c
FIXED[108], FIXED[109] = 'Cm', 'Bb'
for i, c in enumerate(CHORUS[2:]):
    FIXED[110 + i] = c
FIXED.update({84: 'Cm', 122: 'Cm', 123: 'Cm', 124: 'Ab', 125: 'Bb', 126: 'Cm'})
POWER = {'Cm': 'C5', 'Eb': 'Eb5', 'Fm': 'F5', 'F': 'F5', 'Gm': 'G5', 'G': 'G5', 'Ab': 'Ab5', 'Bb': 'Bb5'}


def chord_at(sb, half):
    """Chord name for a source half-bar."""
    if sb in FIXED:
        return FIXED[sb]
    e = chart[f"{sb}.{half}"]
    c = e['chord']
    if c in POWER and e['score'] < 0.60:      # unsure about the third -> drop it
        c = POWER[c]
    return c


ROOT = {k: v[0] for k, v in PCS.items()}

# ---------------------------------------------------------------- music box model
TUNE_C = 6.0


def hz(m):
    return 440.0 * 2 ** ((m - 69 + TUNE_C / 100) / 12)


_cache = {}


def tine(m, damp=1.0):
    """One comb tooth: cantilever partials, pin pluck, natural decay."""
    key = (m, damp)
    if key in _cache:
        return _cache[key]
    f = hz(m)
    dec = float(np.clip(2.4 * (523.0 / f) ** 0.55, 0.7, 3.0)) * damp
    dur = dec * 3.6
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    parts = [(1.0, 1.0, dec), (1.0011, 0.15, dec * 0.85), (6.267, 0.32, dec * 0.18), (17.55, 0.09, dec * 0.05)]
    if f < 200:
        parts.append((2.0, 0.10, dec * 0.4))   # big bass teeth carry a little 2nd partial (lead weights)
    for ratio, amp, d in parts:
        fr = f * ratio
        if fr < SR * 0.45:
            x += amp * np.sin(2 * np.pi * fr * t + RNG.uniform(0, 6.28)) * np.exp(-t / d)
    pluck = butter(noise(n), 3000, 'high', 2) * np.exp(-t / 0.0015) * 0.22
    thunk = butter(noise(n), [350, 1300], 'band', 2) * np.exp(-t / 0.005) * 0.05
    x = (x + pluck + thunk) * np.minimum(1, t / 0.0006)
    k = int(0.08 * SR)
    x[-k:] *= np.linspace(1, 0, k)
    _cache[key] = x
    return x


voices = {k: np.zeros((int((slot_time(TOTAL_BARS * 16) + 14) * SR), 2)) for k in ('mel', 'har', 'arp', 'bass', 'mech')}
N = len(voices['mel'])
EVENTS = []


def play(voice, m, oslot, vel, p=0.0, damp=1.0, jitter=0.003):
    t = slot_time(oslot) + RNG.uniform(-jitter, jitter)
    add_at(voices[voice], pan(tine(int(m), damp) * vel, p), t)
    EVENTS.append((voice, t, int(m)))


def fit(pc, lo, hi):
    """Lowest midi with pitch class pc in [lo, hi]."""
    m = lo + ((pc - lo) % 12)
    return m if m <= hi else m - 12


# ---------------------------------------------------------------- melody
mel = json.load(open(MEL_FILE))
for nt in mel:
    if nt['midi'] < 55:
        nt['midi'] += 12
    while nt['midi'] > 76:
        nt['midi'] -= 12


for nt in mel:
    if nt['midi'] % 12 == 11:
        sb = nt['slot'] // 16
        if chord_at(sb, (nt['slot'] % 16) // 8) not in ('G', 'G5'):
            nt['midi'] += -1 if nt['pf'] - (nt['midi'] - 1) < (nt['midi'] + 1) - nt['pf'] else 1


def long_pcs(a_slot, b_slot):
    """Pitch classes of held melody notes (>= an 8th) sounding in [a_slot, b_slot)."""
    return {nt['midi'] % 12 for nt in mel if nt['len'] >= 2 and nt['slot'] < b_slot and nt['slot'] + nt['len'] > a_slot}


def voicing(ch, avoid):
    """Chord tones (root first) minus any tone a semitone away from a held melody note (root is kept)."""
    pcs = PCS[ch]
    keep = [pcs[0]] + [p for p in pcs[1:] if all(min((p - m) % 12, (m - p) % 12) != 1 for m in avoid)]
    return keep


def melody_in(a, b, limit_slot=None):
    lo, hi = a * 16, (limit_slot if limit_slot is not None else b * 16)
    return [nt for nt in mel if lo <= nt['slot'] < hi]


HOOK = [(0, 'G5'), (4, 'Eb6'), (12, 'Eb6'), (16, 'Eb6'), (18, 'D6'), (24, 'D6'), (28, 'Ab5'), (30, 'G5'),
        (32, 'G5'), (36, 'Ab5'), (38, 'Bb5'), (40, 'Ab5'), (41, 'G5'), (43, 'Ab5'), (45, 'Eb5'), (47, 'D5'),
        (48, 'Eb5'), (50, 'D5'), (52, 'Eb5'), (56, 'G5'), (58, 'C5')]


def arp_tones(pcs, lo=53):
    r = fit(pcs[0], lo, lo + 11)
    if len(pcs) == 1:
        return [r, r + 12, r + 12, r + 24, r + 12, r + 24]
    up = sorted(r + ((p - pcs[0]) % 12) for p in pcs)
    tones = up + [x + 12 for x in up] + [x + 24 for x in up]
    return tones[:6]


ARP_LO = {'verse': 55, 'chorus': 55, 'inter': 55, 'bridge': 50, 'tag': 50, 'final': 55, 'intro': 55}
PATTERNS = {   # (slots between notes, tone indices)
    'verse': (4, [0, 1, 2, 1]),
    'chorus': (2, [0, 2, 3, 2, 4, 2, 3, 2]),
    'inter': (2, [0, 2, 3, 4, 3, 2, 3, 4]),
    'bridge': (2, [3, 2, 1, 2, 3, 2, 1, 0]),
    'final': (2, [0, 2, 3, 4, 3, 2, 3, 4]),
    'tag': (2, [0, 2, 3, 2, 4, 2, 3, 2]),
    'intro': (2, [0, 2, 3, 2, 4, 2, 3, 2]),
}

stats = {'mel_notes': 0, 'chord_tone_long': 0, 'long': 0}
VOICING = {}
for name, a, b, ob in LAYOUT:
    nb = b if a is None else b - a
    for i in range(nb):
        sb = None if a is None else a + i
        oslot0 = (ob + i) * 16
        for half in (0, 1):
            ch = (['Ab', 'Bb', 'Cm', 'Cm'][i]) if a is None else chord_at(sb, half)
            base = oslot0 + half * 8
            vpcs = PCS[ch] if a is None else voicing(ch, long_pcs(sb * 16 + half * 8, sb * 16 + half * 8 + 8))
            VOICING[(name, sb, half)] = vpcs
            # ---- bass: downbeat of each half bar (verses: only when the chord changes)
            prev = None if half == 0 else (['Ab', 'Bb', 'Cm', 'Cm'][i] if a is None else chord_at(sb, 0))
            if name in ('verse', 'bridge') and half == 1 and prev == ch:
                pass
            else:
                bpc = ROOT[ch]
                if a is not None and sb not in FIXED:
                    e = chart[f"{sb}.{half}"]
                    if e['bass_pc'] is not None and e['bass_pc'] in vpcs and not (85 <= sb <= 95):
                        bpc = e['bass_pc']
                play('bass', fit(bpc, 48, 59), base, 0.5 if half == 0 else 0.42, 0.0, damp=0.55)
            # ---- arpeggio
            if name == 'build':
                tones = arp_tones(vpcs, lo=55 + 3 * i)
                for k in range(8):
                    play('arp', tones[[0, 1, 2, 3, 4, 5, 4, 3][k]], base + k, 0.34 + 0.06 * i + 0.006 * (half * 8 + k),
                         0.25, damp=0.6)
            else:
                step, idx = PATTERNS[name]
                tones = arp_tones(vpcs, lo=ARP_LO[name])
                for k, j in enumerate(range(0, 8, step)):
                    ii = idx[(half * (8 // step) + k) % len(idx)]
                    play('arp', tones[ii], base + j, 0.33 if name != 'verse' else 0.30, 0.25, damp=0.55)
            if name == 'final':                 # high sparkle on each half bar
                play('arp', fit(vpcs[-1], 79, 90), base, 0.26, -0.3, damp=0.7)
    # ---- melody (+ harmony voice in the choruses)
    if a is None:
        for p, nn in HOOK:
            play('mel', I.nm(nn), ob * 16 + p, 0.9, -0.08)
        continue
    notes = melody_in(a, b, limit_slot=(a * 16 + 4) if name == 'inter' else None)
    if name == 'build':
        notes = [nt for nt in notes if nt['slot'] >= 107 * 16]   # only the pickup into the final chorus
    for nt in notes:
        oslot = ob * 16 + (nt['slot'] - a * 16)
        m = nt['midi'] + 12
        play('mel', m, oslot, 0.92 + RNG.uniform(-0.05, 0.05), -0.08)
        if name == 'final' and m + 12 <= 98:   # last chorus: melody doubled an octave up (two teeth)
            play('mel', m + 12, oslot, 0.5, 0.1, damp=0.8)
        stats['mel_notes'] += 1
        sb = nt['slot'] // 16
        ch = chord_at(sb, (nt['slot'] % 16) // 8) if sb in FIXED or f"{sb}.0" in chart else None
        if nt['len'] >= 2 and ch:
            stats['long'] += 1
            stats['chord_tone_long'] += (m % 12) in PCS[ch]
            if name in ('chorus', 'final', 'tag'):
                vp = VOICING.get((name, sb, (nt['slot'] % 16) // 8), PCS[ch])
                cands = [c for c in range(m - 9, m - 2) if c % 12 in vp]
                if cands:
                    play('har', max(cands), oslot, 0.5, 0.15, damp=0.8)

print(f"layout: {[(n, a, b, ob) for n, a, b, ob in LAYOUT]}  total bars {TOTAL_BARS}")
print(f"melody notes {stats['mel_notes']}; long notes on chord tones: "
      f"{stats['chord_tone_long']}/{stats['long']} ({stats['chord_tone_long'] / max(1, stats['long']):.0%})")

# ---------------------------------------------------------------- ending: rolled final chord + mechanism
end_slot = TOTAL_BARS * 16
t_end = slot_time(end_slot)
for k, nn in enumerate(['C3', 'G3', 'C4', 'Eb4', 'G4', 'C5', 'Eb5', 'G5']):
    t = t_end + 0.11 * k * (1 + 0.25 * k)
    add_at(voices['arp' if k < 6 else 'mel'], pan(tine(I.nm(nn)) * (0.55 if k < 6 else 0.7), 0.1 * (k - 4) / 4), t)
LAST = t_end + 0.11 * 7 * 2.75
# wind-up at the start, release click, governor whirr while playing, spring stop at the end
for k, t in enumerate([0.15, 0.95, 1.7]):
    add_at(voices['mech'], pan(I.ratchet(n_clicks=[8, 8, 5][k]), 0.1), t, 1.5)
click = I.tick(hi=False)
add_at(voices['mech'], click, PRE - 0.25, 0.35)
t = np.arange(N) / SR
run = ((t > PRE - 0.2) & (t < LAST + 1.2)).astype(float)
run = np.convolve(run, np.hanning(int(0.3 * SR)) / np.hanning(int(0.3 * SR)).sum(), mode='same')
gov_rate = 7.0 * np.where(t < t_end, 1.0, np.clip(1 - (t - t_end) / (LAST + 1.2 - t_end), 0.15, 1))
gov = svf(noise(N), 2600, 1.5, 'bp') * (0.6 + 0.4 * np.sin(2 * np.pi * np.cumsum(gov_rate) / SR)) * run
voices['mech'] += np.stack([gov, gov], axis=1) * 0.004
add_at(voices['mech'], click, LAST + 1.6, 0.3)

# ---------------------------------------------------------------- mix
mix = voices['mel'] * 1.0 + voices['har'] * 0.9 + voices['arp'] * 0.9 + voices['bass'] * 1.0
# wooden case resonance
ti = np.arange(int(0.12 * SR)) / SR
ir = np.zeros_like(ti)
ir[0] = 1.0
for f, tau, a in [(205, 0.03, 0.4), (365, 0.03, 0.6), (585, 0.025, 0.7), (880, 0.018, 0.6), (1340, 0.012, 0.5),
                  (2150, 0.008, 0.4)]:
    ir += 0.012 * a * np.sin(2 * np.pi * f * ti) * np.exp(-ti / tau)
mix = np.stack([ss.oaconvolve(mix[:, c], ir)[:N] for c in range(2)], axis=1)
mix = mix + reverb(mix, room=0.78, damp=0.45, width=1.0) * db(-5)
import pedalboard as pb
from remixlib.dsp import board
mix = board(mix, [pb.HighpassFilter(cutoff_frequency_hz=120), pb.LowShelfFilter(cutoff_frequency_hz=260, gain_db=-4.0),
                  pb.HighShelfFilter(cutoff_frequency_hz=2500, gain_db=3.0)])
mix = mix + voices['mech']
mix = compress(mix, thresh=-20, ratio=1.8, attack=15, release=200)
end = int((LAST + 8.0) * SR)
mix = mix[:end]
k = int(2.5 * SR)
mix[-k:] *= np.linspace(1, 0, k)[:, None] ** 2


def lufs(x):
    tmp = f"{OUT_DIR}/_tmp.wav"
    sf.write(tmp, x.astype(np.float32), SR)
    r = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', tmp, '-af', 'ebur128', '-f', 'null', '-'],
                       capture_output=True, text=True)
    return float([ln for ln in r.stderr.splitlines() if ln.strip().startswith('I:')][-1].split()[1])


TARGET = -14.0
g = TARGET - lufs(mix)
mix, gr = limiter(mix * db(g), ceiling_db=-1.0)
print(f"LUFS {lufs(mix):.2f}, true peak {true_peak_db(mix):.2f} dBTP, max limiter GR {20 * np.log10(gr.min()):.2f} dB, "
      f"length {len(mix) / SR:.1f} s")
sf.write(f"{OUT_DIR}/Shutter_Doll_Music_Box.wav", mix.astype(np.float32), SR, subtype='PCM_24')
json.dump({'layout': [(n, a, b, PRE + ob * BAR) for n, a, b, ob in LAYOUT], 'end': t_end},
          open(f"{OUT_DIR}/_mb_layout.json", 'w'))
np.save(f"{OUT_DIR}/_mb_events.npy", np.array([(['mel', 'har', 'arp', 'bass'].index(v), t, m) for v, t, m in EVENTS
                                              if v != 'mech']))
print("done")
