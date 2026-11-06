"""Shutter Doll (Overclock Mix) — Artcore / J-Core / IDM-glitch rearrangement at 174 BPM.

usage: python3 -I electro.py <scripts_dir> <stems_dir> <melody.json> <chorus_voted.json> <out_dir>

Kept from the song: the lead vocal (time-stretched 145.01 -> 174.01 BPM, pitch kept) and its melody.
Everything else is new: synthesised kicks/basses/supersaws/plucks/glitch FX, plus FluidSynth-rendered
drum kits, piano, music box, strings, orchestra hits and choir (FluidR3 GM).
"""
import os
import sys
import json
import subprocess
import numpy as np
import soundfile as sf
import scipy.signal as ss
import pretty_midi as pm
import pedalboard as pb

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'afterimage'))   # arr_chart
from remixlib.dsp import (SR, db, pan, add_at, svf, butter, polyblep_saw, noise, reverb, delay, compress, limiter,
                          board, true_peak_db)
from remixlib import instruments as I
from arr_chart import parse

STEMS, MELFILE, VOTEDFILE, OUT = sys.argv[2:6]
SF2 = '/usr/share/sounds/sf2/FluidR3_GM.sf2'
RNG = np.random.default_rng(174)

STRETCH = 1.2
SRC_BPM = 145.01
BPM = SRC_BPM * STRETCH
BEAT = 60 / BPM
BAR = 4 * BEAT
S16 = BAR / 16
SRC_BAR = 240 / SRC_BPM
SRC_PH = 0.015
T0 = 0.6
NB = 101


def T(ob):
    return T0 + ob * BAR


def IDX(t):
    return int(round(t * SR))


N = IDX(T(NB) + 7)

# ------------------------------------------------------------------ form & harmony (output bars, C minor)
SECTIONS = [('intro', 0, 16), ('verse', 16, 32), ('drop1', 32, 46), ('kiai', 46, 54), ('idm', 54, 66),
            ('piano', 66, 70), ('build', 70, 74), ('drop2', 74, 93), ('outro', 93, 101)]
VOCAL_SEGMENTS = [(16.25, 46.30, 0), (91.95, 104.0, 38), (107.5, 127.07, 34)]   # (src a, src b, offset): ob = sb - off
HOOKCH = ['Ab', 'Bb', 'Cm', 'Cm']
CHOR = ['Ab', 'Bb', 'Cm', 'Cm', 'Ab', 'Bb', 'Cm', 'Cm', 'Ab', 'Bb', 'Bb', 'Cm', 'Ab', 'Bb']
CH = {}


def put(b, a, c=None):
    CH[b] = [a, c or a]


for b in range(16):
    put(b, HOOKCH[b % 4])
for b, s in {16: 'Abmaj7', 17: 'Abmaj7', 18: ('Abmaj7', 'Eb/G'), 19: 'Gm7', 20: ('Fm9', 'G7sus4'), 21: 'Cm(add9)',
             22: 'Cm/B', 23: 'Cm7/Bb', 24: 'Aø7', 25: 'Abmaj7', 26: ('Fm9', 'Fm6'), 27: ('G7sus4', 'G7b9'),
             28: 'Abmaj7', 29: 'Dbmaj7', 30: 'G7b9b13', 31: 'G7b9b13'}.items():
    put(b, *(s if isinstance(s, tuple) else (s,)))
for i, s in enumerate(CHOR):
    put(32 + i, s)
for i in range(8):
    put(46 + i, HOOKCH[i % 4])
for i, s in enumerate(['Ab(add9)', 'Cm7', 'Bb9sus4', 'Eb6', 'Abmaj9', 'Ebmaj7/G', 'Fm9', 'Dbmaj9', 'Abmaj7', 'Fm7',
                       'Bb7sus4', 'G7sus4']):
    put(54 + i, s)
for i in range(4):
    put(66 + i, HOOKCH[i])
put(70, 'Ab'); put(71, 'Bb'); put(72, 'Cm'); put(73, 'Ab', 'G7#5')
for i, s in enumerate(['Cm', 'Bb'] + CHOR[2:]):
    put(74 + i, s)
for i, s in enumerate(['Cm', 'Cm', 'Ab', 'Bb', 'Cm']):
    put(88 + i, s)
for i in range(4):
    put(93 + i, HOOKCH[i])
put(97, 'Ab'); put(98, 'Bb'); put(99, 'Cm(add9)'); put(100, 'Cm(add9)')


def chord(ob):
    b = int(ob)
    sym = CH[min(b, NB - 1)][1 if ob - b >= 0.5 else 0]
    root, bass, pcs = parse(sym)
    return sym, root, bass, pcs


def section(ob):
    for n, a, b in SECTIONS:
        if a <= ob < b:
            return n
    return 'outro'


def rng_in(pc, lo, hi):
    return [m for m in range(lo, hi + 1) if m % 12 == pc]


def near(pc, target, lo, hi):
    c = rng_in(pc, lo, hi)
    return min(c, key=lambda m: abs(m - target)) if c else None


def voicing(pcs, lo, hi, prev=None, n=4):
    pcs = list(dict.fromkeys(pcs))
    best, bc = None, 1e9
    for rot in range(len(pcs)):
        order = pcs[rot:] + pcs[:rot]
        st = rng_in(order[0], lo, lo + 11)
        if not st:
            continue
        v = [st[0]]
        i = 1
        while len(v) < n:
            p = order[i % len(order)]
            v.append(v[-1] + ((p - v[-1]) % 12 or 12))
            i += 1
        if v[-1] > hi:
            continue
        c = sum(abs(a - b) for a, b in zip(v, sorted(prev))) if prev else abs(np.mean(v) - (lo + hi) / 2)
        if c < bc:
            best, bc = v, c
    return best or []


# ------------------------------------------------------------------ melody (for chord checking + leads)
mel = json.load(open(MELFILE))
for nt in mel:
    while nt['midi'] < 55:
        nt['midi'] += 12
    while nt['midi'] > 76:
        nt['midi'] -= 12
held_ok = held_n = 0
for a, b, off in VOCAL_SEGMENTS:
    for nt in mel:
        sb = nt['slot'] / 16
        if a <= sb < b and nt['len'] >= 2:
            held_n += 1
            held_ok += (nt['midi'] % 12) in chord(sb - off)[3]
print(f"held vocal notes on chord tones: {held_ok}/{held_n} ({held_ok / held_n:.0%})")
voted = json.load(open(VOTEDFILE))
HOOK = [(0, 79), (4, 87), (12, 87), (16, 87), (18, 86), (24, 86), (28, 80), (30, 79), (32, 79), (36, 80), (38, 82),
        (40, 80), (41, 79), (43, 80), (45, 75), (47, 74), (48, 75), (50, 74), (52, 75), (56, 79), (58, 72)]

# ------------------------------------------------------------------ buses & helpers
BUS = {k: np.zeros((N, 2)) for k in ('kick', 'synperc', 'bass', 'reese', 'sub', 'pads', 'lead', 'arps', 'fx', 'chops',
                                     'tex', 'blips')}
KICKS = []


def add(bus, sig, ob, gain=1.0, p=0.0, t_off=0.0):
    s = sig if sig.ndim == 2 else pan(sig, p)
    add_at(BUS[bus], s, T(ob) + t_off, gain)


def hz(m):
    return 440.0 * 2 ** ((m - 69 + 0.06) / 12)


def fadeout(x, sec=0.01):
    k = min(len(x), int(sec * SR))
    x[-k:] *= np.linspace(1, 0, k) if x.ndim == 1 else np.linspace(1, 0, k)[:, None]
    return x


# --- synth voices
def jkick():
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    f = 50 + 240 * np.exp(-t / 0.016) + 900 * np.exp(-t / 0.0015)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.15)
    click = butter(noise(n), 2500, 'high', 2) * np.exp(-t / 0.0025) * 0.5
    x = np.tanh(3.2 * (body + click)) * np.minimum(1, (0.3 - t) / 0.03)
    return x / np.abs(x).max()


JK = jkick()
SUBK = I.kick(dur=0.35, f_end=48, f_start=150, sweep=0.03, drive=1.4, click=0.1, decay=0.12)


def obass(m, dur):
    n = int((dur + 0.02) * SR)
    t = np.arange(n) / SR
    f = hz(m)
    x = polyblep_saw(f, n) + polyblep_saw(f * 1.006, n, 0.4) + 0.5 * np.sign(np.sin(2 * np.pi * f * t))
    x = svf(x, 250 + 2600 * np.exp(-t / 0.045), 1.4, 'lp')
    sub = np.sin(2 * np.pi * f / 2 * t)
    y = np.tanh(1.6 * (0.45 * x + 0.8 * sub)) * np.minimum(1, t / 0.002) * np.exp(-t / (dur * 0.9))
    return fadeout(y, 0.012)


def reese(m, dur, bright=1.0):
    n = int((dur + 0.03) * SR)
    t = np.arange(n) / SR
    f = hz(m)
    x = polyblep_saw(f * 2 ** (0.14 / 12), n, 0.1) + polyblep_saw(f * 2 ** (-0.14 / 12), n, 0.6)
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * (BPM / 60 / 2) * t + RNG.uniform(0, 6))
    x = svf(x, (240 + 520 * lfo) * bright, 1.6, 'lp')
    sub = np.sin(2 * np.pi * f / 2 * t)
    y = np.tanh(1.4 * (0.55 * x + 0.75 * sub)) * np.minimum(1, t / 0.01)
    return fadeout(y, 0.03)


def pluck(m, dur=0.16):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = hz(m)
    x = polyblep_saw(f, n) + 0.6 * np.sign(np.sin(2 * np.pi * f * 1.002 * t))
    x = svf(x, 900 + 7000 * np.exp(-t / 0.035), 1.1, 'lp') * np.exp(-t / 0.07) * np.minimum(1, t / 0.0015)
    return fadeout(x, 0.01)


def lead_note(m, dur):
    s = I.pad_chord([m], dur, attack=0.004, release=0.07, cutoff=6500, voices=7, detune=0.28, bright_env=True)
    sq = 0.18 * np.sign(np.sin(2 * np.pi * hz(m + 12) * np.arange(len(s)) / SR))   # chip layer, octave up
    env = np.minimum(1, np.arange(len(s)) / SR / 0.004) * np.where(np.arange(len(s)) / SR < dur, 1,
                                                                   np.exp(-(np.arange(len(s)) / SR - dur) / 0.03))
    return s + pan(butter(sq, 7000, 'low', 2) * env, 0.0)


def blip(m, dur=0.035):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * hz(m) * t) * np.exp(-t / (dur / 4)) * np.minimum(1, t / 0.0005)


def click():
    n = int(0.012 * SR)
    t = np.arange(n) / SR
    return butter(noise(n), 4000, 'high', 2) * np.exp(-t / 0.0015)


# --- FluidSynth parts
GM = {}


def gm(name, program, drum=False):
    if name not in GM:
        ins = pm.Instrument(program=program, is_drum=drum, name=name)
        if not drum:
            ins.pitch_bends.append(pm.PitchBend(pitch=246, time=0.0))   # +6 cents to match the voice
        GM[name] = ins
    return GM[name]


def NOTE(name, program, m, ob0, ob1, vel, drum=False, jitter=0.0):
    if m is None:
        return
    t0 = T(ob0) + (RNG.uniform(-jitter, jitter) if jitter else 0)
    t1 = max(T(ob1) - 0.005, t0 + 0.03)
    gm(name, program, drum).notes.append(pm.Note(velocity=int(np.clip(vel, 1, 127)), pitch=int(m), start=max(0, t0),
                                                 end=t1))


def DR(kit, note, ob, vel, length=0.12):
    prog = {'room': 8, 'power': 16, '808': 25}[kit]
    NOTE('kit_' + kit, prog, note, ob, ob + length, vel, drum=True)


# ------------------------------------------------------------------ drum programming
def dnb_bar(ob, var, v=1.0):
    kp = {0: [0, 10], 1: [0, 10, 13], 2: [0, 6, 10], 3: [0, 10]}[var]
    for s in kp:
        DR('room', 36, ob + s / 16, 112 * v)
        add('kick', SUBK, ob + s / 16, 0.45 * v)
        KICKS.append(T(ob + s / 16))
    for s in (4, 12):
        DR('room', 38, ob + s / 16, 120 * v)
    for s in {0: [7, 15], 1: [9, 14], 2: [7, 9, 15], 3: []}[var]:
        DR('room', 38, ob + s / 16, 42 * v)
    for s in range(0, 16, 2):
        DR('room', 42, ob + s / 16, (78 if s % 4 == 2 else 58) * v)
    for s in (3, 11):
        DR('room', 42, ob + s / 16, 38 * v)
    if var == 3:
        for k, s in enumerate(range(12, 16)):
            DR('room', 38, ob + s / 16, (70 + 12 * k) * v)
        DR('room', 46, ob + 14 / 16, 70 * v)


def jcore_bar(ob, var, v=1.0, breakcore=False):
    kp = [0, 4, 8, 12] + ([14, 15] if var == 3 else [])
    for s in kp:
        add('kick', JK, ob + s / 16, v * (0.85 if s in (14, 15) else 1.0))
        KICKS.append(T(ob + s / 16))
    for s in (4, 12):
        DR('power', 38, ob + s / 16, 122 * v)
        DR('power', 39, ob + s / 16, 96 * v)
    for s in (2, 6, 10, 14):
        DR('power', 46, ob + s / 16, 82 * v)
    for s in range(16):
        DR('power', 42, ob + s / 16, (62 if s % 2 else 40) * v)
    if breakcore:
        for s in range(16):
            if s not in (4, 12) and RNG.random() < 0.32:
                DR('room', 38, ob + s / 16, int(55 + 50 * RNG.random()))
        if RNG.random() < 0.7:
            s0 = RNG.integers(8, 14)
            for k in range(int(RNG.integers(4, 9))):
                pos = ob + s0 / 16 + k / 32
                if pos < ob + 1:
                    DR('room', 38, pos, 60 + 7 * k)


def idm_bar(ob, i):
    pats = [[0, 6, 11], [0, 3, 10], [0, 7, 9, 14], [0, 5, 10, 13]]
    for s in pats[i % 4]:
        add('kick', SUBK, ob + s / 16, 0.75)
        DR('808', 36, ob + s / 16, 100)
        KICKS.append(T(ob + s / 16))
    DR('808', 38, ob + 8 / 16, 115)
    if i % 2:
        for k in range(3):
            DR('808', 38, ob + 7 / 16 + k / 48, 50 + 15 * k)
    for s in range(16):
        r = RNG.random()
        if r < 0.45:
            DR('808', 42, ob + s / 16, int(50 + 40 * RNG.random()))
        if r < 0.12:                                   # 32nd / 64th bursts
            for k in range(int(RNG.integers(2, 6))):
                DR('808', 42, ob + s / 16 + k / 64, 45 + 10 * k)
    for s in range(32):
        if RNG.random() < 0.18:
            add('synperc', click(), ob + s / 32, 0.6 * RNG.random() + 0.2, RNG.uniform(-0.7, 0.7))
    sym, root, bass, pcs = chord(ob)
    for s in range(16):
        if RNG.random() < 0.28:
            p = pcs[int(RNG.integers(0, len(pcs)))]
            m = near(p, 90, 84, 98)
            add('blips', blip(m), ob + s / 16, 0.5, RNG.uniform(-0.8, 0.8))


# ------------------------------------------------------------------ arrangement
def chord_spans(ob0, ob1):
    """Yield (pos, span_bars, chord) for each chord change between ob0 and ob1."""
    for ob in range(ob0, ob1):
        if CH[ob][0] == CH[ob][1]:
            yield ob, 1.0, chord(ob)
        else:
            yield ob, 0.5, chord(ob)
            yield ob + 0.5, 0.5, chord(ob + 0.5)


prev = {}


def pad_run(ob0, ob1, bus='pads', cutoff=5000, attack=0.01, release=0.15, lo=55, hi=79, gain=1.0, n=4):
    for pos, span, (sym, root, bass, pcs) in chord_spans(ob0, ob1):
        v = voicing(pcs, lo, hi, prev.get(bus), n)
        prev[bus] = v
        add(bus, I.pad_chord(v, span * BAR - 0.01, attack=attack, release=release, cutoff=cutoff), pos, gain)


GATE = [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
gate_curve = np.ones(N)


def gate_section(ob0, ob1, depth=0.85):
    for ob in range(ob0, ob1):
        for s, on in enumerate(GATE):
            if not on:
                a, b = IDX(T(ob + s / 16)), IDX(T(ob + (s + 1) / 16))
                gate_curve[a:b] = 1 - depth


def arps(ob0, ob1, gain=1.0, lo=72, hi=96, pattern=(0, 1, 2, 3, 4, 3, 2, 1)):
    for ob in range(ob0, ob1):
        for s in range(16):
            sym, root, bass, pcs = chord(ob + s / 16)
            tones = sorted(m for p in pcs for m in rng_in(p, lo, hi))[:8]
            m = tones[pattern[s % len(pattern)] % len(tones)]
            add('arps', pluck(m), ob + s / 16, gain * (1.0 if s % 4 == 0 else 0.75), 0.35 if s % 2 else -0.35)


def offbeat_bass(ob0, ob1, gain=1.0):
    for ob in range(ob0, ob1):
        for b4 in range(4):
            pos = ob + b4 / 4 + 1 / 8
            sym, root, bass, pcs = chord(pos)
            add('bass', obass(near(bass, 41, 36, 47), BEAT / 2 - 0.01), pos, gain)


def reese_run(ob0, ob1, gain=1.0, bright=1.0):
    for pos, span, (sym, root, bass, pcs) in chord_spans(ob0, ob1):
        add('reese', reese(near(bass, 41, 36, 47), span * BAR - 0.01, bright), pos, gain)


def piano_arps16(ob0, ob1, vel=52, lo=60, hi=88, up=True):
    for ob in range(ob0, ob1):
        for s in range(16):
            sym, root, bass, pcs = chord(ob + s / 16)
            tones = sorted(m for p in pcs for m in rng_in(p, lo, hi))
            seq = tones + tones[-2:0:-1]
            m = seq[(s + (ob % 2) * 3) % len(seq)] if up else seq[-1 - (s % len(seq))]
            NOTE('piano', 0, m, ob + s / 16, ob + (s + 1) / 16 + 0.05, vel + (10 if s % 4 == 0 else 0), jitter=0.003)
        sym, root, bass, pcs = chord(ob)
        NOTE('piano', 0, near(bass, 41, 36, 47), ob, ob + 1, vel + 6)


def strings_run(ob0, ob1, vel=80, lo=60, hi=86, prog=48, name='strings'):
    for pos, span, (sym, root, bass, pcs) in chord_spans(ob0, ob1):
        v = voicing(pcs, lo, hi, prev.get(name), 4)
        prev[name] = v
        for m in v:
            NOTE(name, prog, m, pos, pos + span + 0.02, vel)


def orch_hit(ob, vel=110):
    sym, root, bass, pcs = chord(ob)
    for m in voicing(pcs, 60, 76, None, 3) + [near(bass, 48, 43, 54)]:
        NOTE('orchhit', 55, m, ob, ob + 0.25, vel)


def crash(ob, vel=110):
    DR('power', 49, ob, vel, 1.0)


def melody_line(notes, ob0, fn, octave=0, slot_limit=None):
    """notes: voted chorus notes (slot within 14 bars)."""
    for nt in notes:
        if slot_limit is not None and nt['slot'] >= slot_limit:
            break
        fn(nt['midi'] + octave, ob0 + nt['slot'] / 16, max(1, min(nt['len'], 4)) / 16)


# ---- INTRO 0-16
for s, m in HOOK:                                   # music box states the hook (the doll)
    NOTE('musicbox', 10, m, s / 16, s / 16 + 0.4, 88)
for s, m in HOOK:
    NOTE('musicbox', 10, m, 4 + s / 16, 4 + s / 16 + 0.4, 70)
pad_run(0, 16, cutoff=1400, attack=0.35, release=0.6, gain=0.55)
piano_arps16(4, 16, vel=46)
for ob in range(8, 14):
    dnb_bar(ob, [0, 1, 2, 3][ob % 4], v=0.6 + 0.05 * (ob - 8))
dnb_bar(14, 0, 0.95)
for k in range(16):                                 # snare roll bar 15
    DR('room', 38, 15 + k / 16, 60 + 4 * k)
reese_run(8, 16, gain=0.7, bright=0.6)
for s, m in HOOK:                                   # supersaw lead takes the hook, filtered (opens via automation)
    add('lead', lead_note(m - 12, 0.25), 8 + s / 16, 0.8)
for s, m in HOOK:
    add('lead', lead_note(m - 12, 0.25), 12 + s / 16, 1.0)
add('fx', I.riser(T(16) - T(13)), 13, 0.5)
add('fx', I.reverse_crash(BAR), 15, 0.5)

# ---- VERSE 16-32 (DnB)
for ob in range(16, 30):
    dnb_bar(ob, [0, 1, 2, 3][ob % 4])
dnb_bar(30, 0)
for k in range(16):
    DR('room', 38, 31 + k / 16, 50 + 4 * k)
for k in range(8):
    DR('room', 38, 30.5 + k / 16, 40 + 3 * k)
reese_run(16, 32)
pad_run(16, 32, cutoff=1900, attack=0.08, release=0.3, gain=0.6)
piano_arps16(16, 32, vel=40, lo=72, hi=96)
arps(24, 32, gain=0.5)
add('fx', I.riser(T(32) - T(30)), 30, 0.55)
add('fx', I.reverse_crash(BAR), 31, 0.55)

# ---- DROP 1 32-46 (J-core)
for ob in range(32, 46):
    jcore_bar(ob, 3 if ob % 4 == 3 else 0)
    if ob % 4 == 0:
        crash(ob)
    if ob % 2 == 0:
        orch_hit(ob)
offbeat_bass(32, 46)
pad_run(32, 46, cutoff=5500, gain=1.0, n=5)
gate_section(32, 46)
arps(32, 46, gain=0.8)
strings_run(32, 46, vel=72, lo=67, hi=91)
for pos, span, (sym, root, bass, pcs) in chord_spans(32, 46):     # piano stabs: 1 and the "and" of 2
    for off in (0, 3 / 8):
        if off < span:
            for m in voicing(pcs, 72, 88, None, 3):
                NOTE('piano', 0, m, pos + off, pos + off + 0.1, 75)
add('fx', I.impact(), 32, 0.6)

# ---- KIAI 46-54: vocal-chop lead + supersaw lead on the chorus melody
for ob in range(46, 54):
    jcore_bar(ob, 3 if ob % 4 == 3 else 0)
    if ob % 4 == 2:
        crash(ob)
    if ob % 2 == 0:
        orch_hit(ob)
crash(46)
offbeat_bass(46, 54)
pad_run(46, 54, cutoff=5500, gain=0.9, n=5)
gate_section(46, 54)
arps(46, 54, gain=0.7)
CHOP_NOTES = []
melody_line(voted, 46, lambda m, pos, d: CHOP_NOTES.append((m, pos, d)), slot_limit=8 * 16)
for m, pos, d in CHOP_NOTES:
    add('lead', lead_note(m + 12, d * BAR), pos, 0.42)

# ---- IDM 54-66
for i, ob in enumerate(range(54, 66)):
    idm_bar(ob, i)
for pos, span, (sym, root, bass, pcs) in chord_spans(54, 66):
    m = near(bass, 41, 36, 47) - 12
    n_ = int((span * BAR) * SR)
    t = np.arange(n_) / SR
    add('sub', np.sin(2 * np.pi * hz(m) * t) * np.minimum(1, t / 0.05) * np.minimum(1, (span * BAR - t) / 0.05), pos,
        0.9)
pad_run(54, 66, cutoff=1100, attack=0.2, release=0.5, gain=0.5)
for pos, span, (sym, root, bass, pcs) in chord_spans(54, 66):    # sparse piano stabs (bit-crushed later)
    for off in (0, 0.375, 0.6875):
        if off < span and RNG.random() < 0.8:
            for m in voicing(pcs, 64, 84, None, 3):
                NOTE('piano_idm', 0, m, pos + off, pos + off + 0.06, 85)

# ---- PIANO BREAKDOWN 66-70 + BUILD 70-74
for s, m in HOOK:
    NOTE('piano', 0, m, 66 + s / 16, 66 + s / 16 + 0.3, 78)
piano_arps16(66, 74, vel=44, lo=48, hi=72)
strings_run(66, 70, vel=64, lo=55, hi=79)
strings_run(70, 74, vel=70, lo=55, hi=84, prog=44, name='tremolo')
for ob in range(70, 73):
    dnb_bar(ob, [0, 1, 2][ob - 70], v=0.55 + 0.12 * (ob - 70))
for k in range(16):
    DR('power', 38, 72 + k / 16, 50 + 3 * k)
for k in range(24):
    DR('power', 38, 73 + k / 32, 80 + k)
reese_run(70, 74, gain=0.7, bright=0.7)
add('fx', I.riser(T(74) - T(70)), 70, 0.6)
add('fx', I.reverse_crash(BAR), 73, 0.55)

# ---- DROP 2 74-93 (J-core x breakcore)
for ob in range(74, 93):
    jcore_bar(ob, 3 if ob % 4 == 1 else 0, breakcore=True)
    if ob % 4 == 2:
        crash(ob)
    if ob % 2 == 0:
        orch_hit(ob, 118)
crash(74, 125)
offbeat_bass(74, 93)
reese_run(74, 93, gain=0.35)
pad_run(74, 93, cutoff=6000, gain=1.0, n=5)
gate_section(74, 93)
arps(74, 93, gain=0.85)
strings_run(74, 93, vel=82, lo=67, hi=91)
strings_run(74, 93, vel=60, lo=55, hi=72, prog=52, name='choir')
for pos, span, (sym, root, bass, pcs) in chord_spans(74, 93):
    for off in (0, 3 / 8):
        if off < span:
            for m in voicing(pcs, 72, 88, None, 3):
                NOTE('piano', 0, m, pos + off, pos + off + 0.1, 82)
add('fx', I.impact(), 74, 0.7)

# ---- OUTRO 93-101: kiai reprise, glitch-out, last chord
for ob in range(93, 97):
    jcore_bar(ob, 3 if ob == 96 else 0)
crash(93, 125)
orch_hit(93, 120)
orch_hit(95, 110)
offbeat_bass(93, 97)
pad_run(93, 99, cutoff=5500, gain=0.9, n=5)
gate_section(93, 99)
arps(93, 99, gain=0.7)
for ob in (97, 98):
    jcore_bar(ob, 0, v=0.8)
offbeat_bass(97, 99)
CHOP2 = []
melody_line(voted, 93, lambda m, pos, d: CHOP2.append((m, pos, d)), slot_limit=4 * 16)
for m, pos, d in CHOP2:
    add('lead', lead_note(m + 12, d * BAR), pos, 0.5)
add('fx', I.impact(), 93, 0.6)
sym, root, bass, pcs = chord(99)
for k, m in enumerate([near(bass, 41, 36, 47)] + voicing(pcs, 60, 84, None, 5)):
    NOTE('piano', 0, m, 99 + k * 0.012, 101.5, 70)
NOTE('musicbox', 10, 84, 99.25, 101.5, 80)          # the doll has the last word: C6
NOTE('musicbox', 10, 79, 99.75, 101.5, 60)

# ------------------------------------------------------------------ render FluidSynth parts
def render(names, tag, gain=0.45):
    mid = pm.PrettyMIDI(resolution=960, initial_tempo=120)
    for nm_ in names:
        if nm_ in GM:
            mid.instruments.append(GM[nm_])
    path, wav = f"{OUT}/{tag}.mid", f"{OUT}/{tag}.wav"
    mid.write(path)
    subprocess.run(['fluidsynth', '-ni', '-q', '-R', '0', '-C', '0', '-g', str(gain), '-r', str(SR), '-F', wav, SF2,
                    path], check=True, capture_output=True)
    x, _ = sf.read(wav, dtype='float64')
    y = np.zeros((N, 2))
    y[:min(N, len(x))] = x[:N]
    return y


DRUMS = render(['kit_room'], 'drums_room') + render(['kit_power'], 'drums_power')   # one kit per file (shared ch.10)
IDMDR = render(['kit_808'], 'idm_drums')
KEYS = render(['piano', 'musicbox'], 'keys')
PIDM = render(['piano_idm'], 'piano_idm')
ORCH = render(['strings', 'tremolo', 'choir', 'orchhit'], 'orch')
print("GM notes:", {k: len(v.notes) for k, v in GM.items()})

# ------------------------------------------------------------------ vocal: stretch to 174 BPM, cut, place
vsrc, _ = sf.read(f"{STEMS}/04_shutter_doll_vocals.wav", dtype='float32')
import librosa


def onset_lag(orig, mod, ratio):
    def ons(x):
        y = librosa.resample(x.mean(1).astype(np.float64), orig_sr=SR, target_sr=22050)
        return librosa.onset.onset_detect(y=y, sr=22050, hop_length=64, units='time')
    o1, o2 = ons(orig), ons(mod)
    dd = [o2[np.argmin(np.abs(o2 - t))] - t for t in o1 / ratio if len(o2) and np.min(np.abs(o2 - t)) < 0.15]
    return float(np.median(dd)) if dd else 0.0


def src_t(sb):
    return SRC_PH + sb * SRC_BAR


VOX = np.zeros((N, 2))
venv = np.sqrt(np.maximum(ss.oaconvolve((vsrc.astype(np.float64) ** 2).mean(1), np.ones(882) / 882, mode='same'), 0))


def quiet(sb0, sb1):
    a, b = IDX(src_t(sb0)), IDX(min(src_t(sb1), len(vsrc) / SR))
    return (a + int(np.argmin(venv[a:b]))) / SR


for a, b, off in VOCAL_SEGMENTS:
    t_in = quiet(a - 0.06, a + 0.06)
    t_out = quiet(b - 0.12, b + 0.2) if b < 127 else len(vsrc) / SR
    pad_s = 0.3
    raw = vsrc[max(0, IDX(t_in - pad_s)):IDX(t_out + pad_s)]
    st = pb.time_stretch(raw.T.copy(), SR, stretch_factor=STRETCH, high_quality=True, transient_mode='crisp',
                         preserve_formants=True).T.astype(np.float64)
    ctl = ss.resample_poly(raw, 5, 6, axis=0)
    lag = onset_lag(raw, st, STRETCH) - onset_lag(raw, ctl, STRETCH)   # detector bias removed
    lead_in = (t_in - max(0, t_in - pad_s)) / STRETCH
    x = st[IDX(lead_in + lag):IDX(lead_in + lag + (t_out - t_in) / STRETCH)].copy()
    print(f"vocal segment src {a}-{b}: stretch lag {lag * 1000:+.1f} ms (compensated)")
    fi, fo = int(0.02 * SR), int(0.05 * SR)
    x[:fi] *= np.linspace(0, 1, fi)[:, None]
    x[-fo:] *= np.linspace(1, 0, fo)[:, None]
    dest = T(a - off) + (t_in - src_t(a)) / STRETCH
    i = IDX(dest)
    VOX[i:i + len(x)] += x[:N - i]
# downward expander against separation bleed
e = np.sqrt(np.maximum(ss.oaconvolve((VOX ** 2).mean(1), np.ones(441) / 441, mode='same'), 0))
g = np.clip(e / db(-44), 0, 1) ** 1.5
g = ss.oaconvolve(g, np.hanning(2646) / np.hanning(2646).sum(), mode='same')
VOX *= np.clip(g, 0, 1)[:, None]


def ob_of_src(sb, off):
    return sb - off


# vocal glitches in the IDM section: chop the long C5 (src 99) and stutter the bridge's last word
def stutter(x, ob0, schedule, src_ob=None):
    """Overwrite x from ob0 with repeats of the audio found at src_ob (default ob0).
    schedule: list of (slice_len_in_bars, repeats[, speed])."""
    s0 = IDX(T(src_ob if src_ob is not None else ob0))
    pos = IDX(T(ob0))
    need = max(int(IDX(it[0] * BAR) * ((it[2] if len(it) > 2 else 1.0) ** it[1])) for it in schedule) + 10
    src = x[s0:s0 + need].copy()
    for item in schedule:
        L, reps = item[0], item[1]
        speed = item[2] if len(item) > 2 else 1.0
        n = IDX(L * BAR)
        for r in range(reps):
            sp = speed ** r if speed != 1.0 else 1.0
            src_len = int(n * sp)
            seg = src[:src_len].copy()
            if sp != 1.0:
                idx = np.linspace(0, len(seg) - 1, n)
                seg = np.stack([np.interp(idx, np.arange(len(seg)), seg[:, c]) for c in range(2)], axis=1)
            seg = seg[:n]
            k = min(len(seg) // 4, int(0.002 * SR))
            if k > 0:
                seg[:k] *= np.linspace(0, 1, k)[:, None]
                seg[-k:] *= np.linspace(1, 0, k)[:, None]
            x[pos:pos + len(seg)] = seg
            pos += n
    return pos


def gate_chop(x, ob0, ob1, step=1 / 32, duty=0.5):
    for k in range(int(round((ob1 - ob0) / step))):
        a = IDX(T(ob0 + k * step + step * duty))
        b = IDX(T(ob0 + (k + 1) * step))
        f = min(int(0.002 * SR), (b - a) // 2)
        x[a:b] *= 0.0
        if f:
            x[a - f:a] *= np.linspace(1, 0, f)[:, None]


gate_chop(VOX, 99 - 38 + 0.6, 99 - 38 + 1.2)           # second half of the long C5: 32nd chopping
stutter(VOX, 103 - 38 + 0.6875, [(1 / 16, 3), (1 / 32, 4)], src_ob=103 - 38 + 0.5)
VOX[IDX(T(103 - 38 + 0.6875 + 3 / 16 + 4 / 32)):IDX(T(66))] *= 0

# ------------------------------------------------------------------ vocal-chop lead (kiai & outro)
SRCV = {'C5': (164.15, 165.75, 72.10), 'F4': (169.45, 171.70, 65.10)}
shifted = {}


def chop_src(name, semis):
    key = (name, round(semis, 2))
    if key not in shifted:
        a, b, _ = SRCV[name]
        seg = vsrc[IDX(a):IDX(b)].copy()
        y = pb.time_stretch(seg.T.copy(), SR, stretch_factor=1.0, pitch_shift_in_semitones=float(semis),
                            high_quality=True, preserve_formants=True).T.astype(np.float64)
        shifted[key] = y
    return shifted[key]


CHOP_ERR = []


def vocal_chop(m, ob, dur_bars, gain=1.0):
    name = 'C5'   # the F4 take has almost no fundamental in the separated stem (H1 < H3 by > 50 dB)
    y = chop_src(name, m + 0.06 - SRCV[name][2])
    probe = y[IDX(0.1):IDX(0.4)].mean(1)
    X = np.abs(np.fft.rfft(probe * np.hanning(len(probe))))
    fr = np.fft.rfftfreq(len(probe), 1 / SR)
    band = (fr > 150) & (fr < 3000)
    fpk = fr[band][np.argmax(X[band])]
    CHOP_ERR.append(1200 * np.log2(fpk / hz(m)))
    n = min(len(y) - IDX(0.06), IDX(dur_bars * BAR * 0.92))
    s = y[IDX(0.06):IDX(0.06) + n].copy()
    k1, k2 = int(0.004 * SR), min(n // 3, int(0.03 * SR))
    s[:k1] *= np.linspace(0, 1, k1)[:, None]
    s[-k2:] *= np.linspace(1, 0, k2)[:, None]
    add('chops', s, ob, gain)
    add('chops', s[:, ::-1] * 0.6, ob, gain, t_off=0.011)


for m, pos, d in CHOP_NOTES + CHOP2:
    vocal_chop(m + 12, pos, d)                      # chops an octave up: classic J-core vocal-chop register
ce = np.array(CHOP_ERR)
print(f"vocal-chop pitch check (strongest partial vs target): median {np.median(ce):+.0f} cents, "
      f"max |err| {np.abs(ce).max():.0f} cents over {len(ce)} chops")
# frozen vocal texture (granular C5) for the intro, IDM section and piano breakdown
gsrc = vsrc[IDX(164.3):IDX(165.6)].mean(1).astype(np.float64)


def granular(dur, density=70, grain=0.07):
    n = int(dur * SR)
    out = np.zeros((n, 2))
    gl = int(grain * SR)
    w = np.hanning(gl)
    for _ in range(int(density * dur)):
        p = RNG.integers(0, len(gsrc) - gl)
        t = RNG.integers(0, n - gl)
        a = (RNG.uniform(-1, 1) + 1) * np.pi / 4
        g_ = gsrc[p:p + gl] * w
        out[t:t + gl, 0] += g_ * np.cos(a)
        out[t:t + gl, 1] += g_ * np.sin(a)
    ramp = np.minimum(1, np.minimum(np.arange(n) / (0.8 * SR), (n - np.arange(n)) / (1.0 * SR)))
    return butter(out, 4500, 'low', 2) * ramp[:, None]


for a, b, gn in [(2, 8, 0.6), (54, 66, 0.35), (66, 74, 0.5)]:
    add('tex', granular(T(b) - T(a)), a, gn)

# ------------------------------------------------------------------ bus processing
t_ax = np.arange(N) / SR


def sidechain(times, depth, tau):
    gcur = np.ones(N)
    for tk in times:
        i0 = IDX(tk)
        L = int(5 * tau * SR)
        seg = np.arange(L) / SR
        shape = 1 - depth * np.exp(-seg / tau) * (1 - np.exp(-seg / 0.002))
        gcur[i0:i0 + L] = np.minimum(gcur[i0:i0 + L], shape[:max(0, min(L, N - i0))])
    return gcur


def rms(x, a, b):
    s = x[IDX(T(a)):IDX(T(b))]
    return 20 * np.log10(np.sqrt(np.mean(s ** 2)) + 1e-12)


def lvl(x, target, a, b):
    return x * db(target - rms(x, a, b))


SC_B = sidechain(KICKS, 0.9, 0.07)
SC_P = sidechain(KICKS, 0.6, 0.11)
SC_L = sidechain(KICKS, 0.3, 0.09)

# intro lead: low-pass opening 8 -> 16
lp = np.exp(np.interp(t_ax, [T(8), T(15.9), T(16)], np.log([500, 9000, 20000])))
seg = slice(IDX(T(7.9)), IDX(T(16.1)))
BUS['lead'][seg] = svf(BUS['lead'][seg], lp[seg], 0.8, 'lp')
# intro drums: low-pass opening 8 -> 15, build drums/bass: high-pass sweep into the drops
DR_ALL = DRUMS.copy()
dl = np.exp(np.interp(t_ax, [T(8), T(14.5), T(15)], np.log([350, 6000, 20000])))
seg = slice(IDX(T(7.9)), IDX(T(15.1)))
DR_ALL[seg] = svf(DR_ALL[seg], dl[seg], 0.8, 'lp')
dl2 = np.exp(np.interp(t_ax, [T(70), T(72.5), T(73)], np.log([400, 5000, 20000])))
seg = slice(IDX(T(69.9)), IDX(T(73.1)))
DR_ALL[seg] = svf(DR_ALL[seg], dl2[seg], 0.8, 'lp')
for a, b in [(30, 32), (72, 74)]:
    hpv = np.exp(np.interp(t_ax, [T(a), T(b) - 0.01, T(b)], np.log([20, 450, 20])))
    seg = slice(IDX(T(a - 0.1)), IDX(T(b + 0.05)))
    for bb in ('bass', 'reese'):
        BUS[bb][seg] = svf(svf(BUS[bb][seg], hpv[seg], 0.7, 'hp'), hpv[seg], 0.7, 'hp')
    BUS['pads'][seg] = svf(BUS['pads'][seg], hpv[seg], 0.7, 'hp')

# references: vocal level in drop 1
V_REF = rms(VOX, 32, 46)
KICK = lvl(BUS['kick'], V_REF + 1.0, 32, 46)
DRM = lvl(board(DR_ALL, [pb.Compressor(threshold_db=-20, ratio=3, attack_ms=3, release_ms=80),
                         pb.Distortion(drive_db=4)]), V_REF - 2.0, 32, 46)
DRM = DRM + reverb(DRM, room=0.35, damp=0.6) * db(-16)
CLIP = 0.62
DRM = np.tanh(DRM / CLIP) * CLIP            # soft-clip the break/snare bus (density, less master limiting)
IDMD = lvl(board(IDMDR, [pb.Compressor(threshold_db=-18, ratio=3), pb.Bitcrush(bit_depth=10)]), V_REF - 3.0, 54, 66)
def peak_to(x, pk):          # sparse transients: set by peak, not RMS
    return x * (pk / (np.abs(x).max() + 1e-12))


SYNP = peak_to(BUS['synperc'], 0.16)
BLIP = peak_to(BUS['blips'] + delay(BUS['blips'], 3 * S16, 0.35, 600, 9000) * db(-6), 0.14)
BASS = lvl(BUS['bass'] * SC_B[:, None], V_REF - 1.5, 32, 46)
BASS = BASS + lvl(BUS['reese'] * sidechain(KICKS, 0.5, 0.08)[:, None], V_REF - 5.0, 16, 30)
BASS = BASS + lvl(BUS['sub'], V_REF - 7.0, 54, 66)
PADS = BUS['pads'] * (SC_P * gate_curve)[:, None]
PADS = lvl(PADS + reverb(PADS, room=0.75, damp=0.5) * db(-6), V_REF - 8.0, 32, 46)
ARPS = BUS['arps'] * SC_L[:, None]
ARPS = lvl(ARPS + delay(ARPS, 3 * S16, 0.4, 800, 9000) * db(-8) + reverb(ARPS, 0.7) * db(-10), V_REF - 12.0, 32, 46)
LEAD = BUS['lead'] * SC_L[:, None]
LEAD = lvl(LEAD + reverb(LEAD, room=0.7) * db(-10) + delay(LEAD, 3 * S16, 0.3, 600, 7000) * db(-12), V_REF - 6.0, 46, 54)
CHOPS = board(BUS['chops'], [pb.Compressor(threshold_db=-20, ratio=3), pb.HighpassFilter(cutoff_frequency_hz=180)])
CHOPS = lvl(CHOPS + reverb(CHOPS, room=0.7) * db(-8) + delay(CHOPS, 2 * S16, 0.35, 500, 8000) * db(-10), V_REF, 46, 54)
TEX = lvl(BUS['tex'] + reverb(BUS['tex'], room=0.9) * db(-2), V_REF - 15.0, 2, 8)
KEYS_ = KEYS * (1 - 0.2 * (1 - SC_L))[:, None]
KEYS_ = lvl(KEYS_ + reverb(KEYS_, room=0.8, damp=0.45) * db(-7), V_REF - 10.0, 16, 30)
PI = board(PIDM, [pb.Bitcrush(bit_depth=6)])
PI = lvl(PI + reverb(PI, room=0.7) * db(-6), V_REF - 12.0, 54, 66)
ORC = ORCH * (1 - 0.3 * (1 - SC_L))[:, None]
ORC = lvl(ORC + reverb(ORC, room=0.85, damp=0.4) * db(-5), V_REF - 9.0, 32, 46)
FX = BUS['fx'] * db(V_REF + 8.0)
VOXP = board(VOX, [pb.HighpassFilter(cutoff_frequency_hz=120), pb.PeakFilter(cutoff_frequency_hz=3500, gain_db=2.0, q=0.7),
                   pb.HighShelfFilter(cutoff_frequency_hz=10000, gain_db=2.0),
                   pb.Compressor(threshold_db=-22, ratio=3.5, attack_ms=4, release_ms=90)])
VOXP = VOXP * db(V_REF - rms(VOXP, 32, 46))
VOXP *= db(1.5)
VOXP, _ = limiter(VOXP, ceiling_db=rms(VOXP, 32, 46) + 11.0, lookahead=0.003, release=0.12)   # tame belted peaks
VOX_ALL = VOXP + reverb(VOXP, room=0.6, damp=0.5, predelay=0.02) * db(-13)

MUSIC = KICK + DRM + IDMD + SYNP + BLIP + BASS + PADS + ARPS + LEAD + CHOPS + TEX + KEYS_ + PI + ORC + FX
# section faders: a soft intro that grows as the drums arrive, and a slightly lower IDM section
fader = np.interp(t_ax, [0, T(7.9), T(8), T(12), T(15.5), T(54), T(54.05), T(65.9), T(66)],
                  [db(-9), db(-9), db(-6), db(-3.5), 1.0, 1.0, db(-3), db(-3), 1.0])
MUSIC *= fader[:, None]

# ------------------------------------------------------------------ glitch edits on the music bus
def silence(x, ob0, ob1):
    a, b = IDX(T(ob0)), IDX(T(ob1))
    f = int(0.004 * SR)
    x[a:b] = 0
    x[a - f:a] *= np.linspace(1, 0, f)[:, None]


def tape_stop_at(x, ob0, ob1, curve=1.7):
    a, b = IDX(T(ob0)), IDX(T(ob1))
    n = b - a
    u = np.arange(n) / n
    pos = a + np.cumsum((1 - u) ** curve)
    seg = np.stack([np.interp(pos, np.arange(len(x)), x[:, c]) for c in range(2)], axis=1)
    seg[-int(0.02 * SR):] *= np.linspace(1, 0, int(0.02 * SR))[:, None]
    x[a:b] = seg


def crush(x, ob0, ob1, bits=5, ds=6):
    a, b = IDX(T(ob0)), IDX(T(ob1))
    seg = x[a:b]
    held = np.repeat(seg[::ds], ds, axis=0)[:len(seg)]
    q = 2 ** (bits - 1)
    x[a:b] = np.round(held * q) / q


def shuffle16(x, ob, swaps=4):
    base = x[IDX(T(ob)):IDX(T(ob + 1))].copy()
    sl = [base[IDX(k * S16):IDX((k + 1) * S16)] for k in range(16)]
    order = list(range(16))
    for _ in range(swaps):
        i, j = RNG.integers(0, 16, 2)
        order[i], order[j] = order[j], order[i]
    pos = IDX(T(ob))
    for k in order:
        s = sl[k].copy()
        f = min(len(s) // 4, int(0.0015 * SR))
        s[:f] *= np.linspace(0, 1, f)[:, None]
        s[-f:] *= np.linspace(1, 0, f)[:, None]
        x[pos:pos + len(s)] = s
        pos += len(s)


stutter(MUSIC, 7.5, [(1 / 16, 4), (1 / 32, 4)])
end = stutter(MUSIC, 15.5, [(1 / 16, 2), (1 / 32, 4), (1 / 64, 6)])
silence(MUSIC, 15.5 + 1 / 8 + 1 / 8 + 6 / 64, 16)
for ob in (19.75, 23.75, 27.75):
    stutter(MUSIC, ob, [(1 / 32, 8)])
silence(MUSIC, 31.875, 32)
for ob in (35.75, 39.75):
    stutter(MUSIC, ob, [(1 / 16, 2), (1 / 32, 4)])
stutter(MUSIC, 43.5, [(1 / 8, 2), (1 / 16, 2), (1 / 32, 4)])
stutter(MUSIC, 49.5, [(1 / 16, 8, 1.06)])                # rising-pitch stutter
tape_stop_at(MUSIC, 53.25, 54)
for ob in (56, 58, 60, 62, 63, 64):
    shuffle16(MUSIC, ob, swaps=3 + (ob - 56) // 2)
for ob in (57.75, 61.75, 63.5):
    stutter(MUSIC, ob, [(1 / 32, 4), (1 / 64, 8)])
stutter(MUSIC, 65.5, [(1 / 32, 8, 1.08)])
crush(MUSIC, 64.5, 65.5, bits=6, ds=4)
stutter(MUSIC, 73.0, [(1 / 16, 4), (1 / 32, 4), (1 / 64, 8)])
silence(MUSIC, 73.75, 74)
for ob in (77.75, 81.75, 85.75):
    stutter(MUSIC, ob, [(1 / 32, 8)])
stutter(MUSIC, 89.5, [(1 / 16, 2), (1 / 32, 4)])
crush(MUSIC, 89.75, 90, bits=4, ds=8)
silence(MUSIC, 92.875, 93)
stutter(MUSIC, 97.0, [(1 / 8, 4)])
stutter(MUSIC, 97.5, [(1 / 16, 8, 1.07)])
crush(MUSIC, 98.0, 98.5, bits=5, ds=6)
stutter(MUSIC, 98.0, [(1 / 32, 16, 1.03)])
tape_stop_at(MUSIC, 98.5, 99)

# ------------------------------------------------------------------ master
mix = MUSIC + VOX_ALL
mix = compress(mix, thresh=-16, ratio=2.0, attack=10, release=120)
end_t = T(101.5) + 2.5
mix = mix[:IDX(end_t)]
k = int(2.0 * SR)
mix[-k:] *= np.linspace(1, 0, k)[:, None] ** 2


def lufs(x):
    tmp = f"{OUT}/_tmp.wav"
    sf.write(tmp, x.astype(np.float32), SR)
    r = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', tmp, '-af', 'ebur128', '-f', 'null', '-'],
                       capture_output=True, text=True)
    return float([ln for ln in r.stderr.splitlines() if ln.strip().startswith('I:')][-1].split()[1])


TARGET = -8.5
gdb = TARGET - lufs(mix)
for _ in range(4):
    out, gr = limiter(mix * db(gdb), ceiling_db=-1.0)
    L = lufs(out)
    if abs(L - TARGET) < 0.15:
        break
    gdb += TARGET - L
mix = out
gr_db = 20 * np.log10(gr)
order = np.argsort(gr_db)
picked = []
for i in order:
    if all(abs(i - j) > SR for j in picked):
        picked.append(i)
    if len(picked) == 5:
        break
for i in picked[:3]:
    w = slice(i - IDX(0.03), i + IDX(0.03))
    parts = dict(KICK=KICK, DRM=DRM, IDMD=IDMD, SYNP=SYNP, BLIP=BLIP, BASS=BASS, PADS=PADS, ARPS=ARPS, LEAD=LEAD,
                 CHOPS=CHOPS, TEX=TEX, KEYS=KEYS_, PI=PI, ORC=ORC, FX=FX, VOX=VOX_ALL)
    pk = {k: round(float(np.abs(v[w]).max()), 3) for k, v in parts.items()}
    print(f"  bar {(i / SR - T0) / BAR:.2f} pre-master peaks:", {k: v for k, v in sorted(pk.items(), key=lambda kv: -kv[1])[:6]},
          "| music after edits", round(float(np.abs(MUSIC[w]).max()), 3))
print("heaviest limiting at bars:", [(round((i / SR - T0) / BAR, 2), round(float(gr_db[i]), 1)) for i in picked])
print(f"LUFS {lufs(mix):.2f}, true peak {true_peak_db(mix):.2f} dBTP, max GR {20 * np.log10(gr.min()):.2f} dB, "
      f"mean GR in drops {20 * np.log10(gr[IDX(T(32)):IDX(T(46))].mean()):.2f} dB, length {len(mix) / SR:.1f} s")
for n_, a, b in SECTIONS:
    print(f"  {n_:6s} {T(a):6.1f}-{T(b):6.1f}s  mix {rms(mix, a, b):6.1f}  voice {rms(VOX_ALL, a, b):6.1f}  "
          f"kick {rms(KICK, a, b):6.1f}  drums {rms(DRM + IDMD, a, b):6.1f}  bass {rms(BASS, a, b):6.1f}  "
          f"pads {rms(PADS, a, b):6.1f}  lead {rms(LEAD + CHOPS, a, b):6.1f}  keys {rms(KEYS_ + PI, a, b):6.1f}  "
          f"orch {rms(ORC, a, b):6.1f}")
# --- checks: vocal timing vs the 174 BPM grid, and the pitch of the vocal-chop lead
import librosa
vy = librosa.resample(VOX.mean(1), orig_sr=SR, target_sr=22050)
von = librosa.onset.onset_detect(y=vy, sr=22050, hop_length=64, units='time')
for (a, b, off) in VOCAL_SEGMENTS:
    sel = von[(von >= T(a - off)) & (von < T(b - off))]
    z = np.mean(np.exp(2j * np.pi * (sel - T0) / S16))
    print(f"  vocal onsets vs grid, segment src {a}-{b}: circular mean {np.angle(z) / (2 * np.pi) * S16 * 1000:+.1f} ms "
          f"(R={abs(z):.2f}, n={len(sel)})")
sf.write(f"{OUT}/Shutter_Doll_Overclock.wav", mix.astype(np.float32), SR, subtype='PCM_24')
json.dump({n_: [T(a), T(b)] for n_, a, b in SECTIONS}, open(f"{OUT}/_sections.json", 'w'))
print("done")
