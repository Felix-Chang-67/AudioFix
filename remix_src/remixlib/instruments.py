"""Synthesized instruments & sound effects for the Clockwork Darkroom remix."""
import numpy as np
from .dsp import (SR, RNG, svf, butter, polyblep_saw, sine, noise, env_ad, pan, reverb, db)

TUNE_CENTS = 6.0


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69 + TUNE_CENTS / 100) / 12)


NOTE = {n: i for i, n in enumerate(['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'])}


def nm(s):
    """'Eb5' -> midi number."""
    name, octv = s[:-1], int(s[-1])
    return 12 * (octv + 1) + NOTE[name]


def _norm(x, peak=1.0, tail=0.015):
    x = np.array(x, dtype=np.float64)
    k = min(len(x), int(tail * SR))
    if k > 1:
        fade = np.linspace(1, 0, k)
        x[-k:] *= fade if x.ndim == 1 else fade[:, None]
    return x / (np.abs(x).max() + 1e-12) * peak


# ---------------------------------------------------------------- drums
def kick(dur=0.5, f_end=49.0, f_start=190.0, sweep=0.032, drive=1.8, click=0.25, decay=0.17):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f_end + (f_start - f_end) * np.exp(-t / sweep)
    body = sine(f, n) * env_ad(n, 0.0015, decay)
    cl = butter(noise(n), 2500, 'high', 2) * env_ad(n, 0.0002, 0.004) * click
    x = np.tanh(drive * (body + cl)) / np.tanh(drive)
    x *= np.minimum(1, (dur - t) / 0.02)  # end fade
    return _norm(x)


def industrial_kick():
    k = kick(dur=0.3, f_end=52, f_start=240, sweep=0.02, drive=3.0, click=0.5, decay=0.07)
    return _norm(butter(k, 9000, 'low', 2))


def snare(dur=0.45, tone=188.0):
    n = int(dur * SR)
    body = sine(tone * (1 + 0.5 * np.exp(-np.arange(n) / SR / 0.01)), n) * env_ad(n, 0.0008, 0.06)
    nz = butter(noise(n), [1200, 9000], 'band', 2) * env_ad(n, 0.0005, 0.16)
    return _norm(np.tanh(1.5 * (0.7 * body + 0.9 * nz)))


def clap(dur=0.5):
    n = int(dur * SR)
    nz = butter(noise(n), [900, 5200], 'band', 2)
    e = np.zeros(n)
    t = np.arange(n) / SR
    for k, off in enumerate([0.0, 0.009, 0.019, 0.028]):
        tt = t - off
        dec = 0.005 if k < 3 else 0.13
        e += np.where(tt >= 0, np.exp(-np.maximum(tt, 0) / dec), 0) * (0.8 if k < 3 else 1.0)
    return _norm(nz * e)


def big_snare():
    """Halftime backbeat: clap + snare layer."""
    s = snare()
    c = clap()
    n = max(len(s), len(c))
    x = np.zeros(n)
    x[:len(s)] += 0.75 * s
    x[:len(c)] += 0.8 * c
    return _norm(x)


_HAT_FREQS = [205.3, 304.4, 369.6, 522.7, 540.0, 800.0]


def hat(dur=0.05, open_=False):
    d = 0.32 if open_ else dur
    n = int((d + 0.05) * SR)
    t = np.arange(n) / SR
    metal = sum(np.sign(np.sin(2 * np.pi * f * 1.73 * t + RNG.uniform(0, 6.28))) for f in _HAT_FREQS)
    x = 0.6 * metal + noise(n)
    x = butter(x, [6800, 15000], 'band', 2)
    x *= env_ad(n, 0.0004, d / 2.5)
    return _norm(x)


# ---------------------------------------------------------------- mechanical sounds
def _resonant_click(modes, n_dur=0.08, burst=0.0018, gain=1.0):
    n = int(n_dur * SR)
    exc = noise(n) * env_ad(n, 0.0001, burst)
    out = np.zeros(n)
    for f, q, a in modes:
        out += a * svf(exc, f, q, 'bp')
    out += 0.25 * butter(exc, 3000, 'high', 2)
    return out * gain


def shutter(full=True):
    """Camera shutter 'ka-chak' (full) or a short shutter 'tick' for hi-hat use."""
    a = _resonant_click([(1850, 12, 1.0), (3150, 14, 0.8), (4700, 16, 0.6), (7300, 10, 0.5)], 0.07)
    thump = sine(140, len(a)) * env_ad(len(a), 0.0005, 0.008) * 0.6
    a = a + thump
    if not full:
        return _norm(a * env_ad(len(a), 0.0001, 0.02))
    b = _resonant_click([(2250, 12, 0.8), (3900, 14, 0.7), (6100, 12, 0.5), (9100, 8, 0.4)], 0.09) * 0.75
    n = int(0.25 * SR)
    x = np.zeros(n)
    x[:len(a)] += a
    off = int(0.062 * SR)
    x[off:off + len(b)] += b
    for k in range(4):  # tiny mechanical rattle after the second curtain
        o = off + int((0.012 + 0.007 * k) * SR)
        r = _resonant_click([(5200 + 400 * k, 10, 0.5)], 0.02) * (0.25 / (k + 1))
        x[o:o + len(r)] += r
    return _norm(x)


def film_advance(dur=0.45):
    """Motor-drive / film advance whirr."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 95 + 60 * np.sin(np.pi * t / dur)
    buzz = polyblep_saw(f, n) * 0.5 + noise(n) * 0.5
    x = svf(buzz, 1800 + 900 * np.sin(np.pi * t / dur), 2.5, 'bp')
    x *= np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 0.6
    return _norm(x)


def ratchet(n_clicks=11, rate=13.0, jitter=0.12):
    """Music box winding key: a burst of ratchet clicks."""
    dur = n_clicks / rate + 0.1
    x = np.zeros(int(dur * SR))
    t = 0.0
    for k in range(n_clicks):
        c = _resonant_click([(2600, 9, 1.0), (4100, 10, 0.7), (6400, 8, 0.45), (900, 6, 0.4)], 0.05, 0.0012)
        g = 0.65 + 0.35 * RNG.random()
        i = int(t * SR)
        x[i:i + len(c)] += c[:len(x) - i] * g
        t += (1 / rate) * (1 + jitter * (RNG.random() - 0.5))
    return _norm(x)


def tick(hi=True):
    f = (3300, 5200) if hi else (2500, 4300)
    x = _resonant_click([(f[0], 18, 1.0), (f[1], 20, 0.6)], 0.05, 0.0008)
    return _norm(x)


# ---------------------------------------------------------------- tonal
def music_box_note(m, vel=1.0, dur=None):
    """Music-box tine: cantilever-beam partials (1, 6.27, 17.55) with fast upper decay."""
    f = midi_hz(m)
    base_dec = 2.2 * (523.0 / f) ** 0.6
    dur = dur or min(4.0, base_dec * 3.2)
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for ratio, amp, dec in [(1.0, 1.0, base_dec), (1.0012, 0.18, base_dec * 0.8),
                            (6.267, 0.22, base_dec * 0.12), (17.55, 0.06, base_dec * 0.035)]:
        fr = f * ratio
        if fr < SR * 0.45:
            x += amp * np.sin(2 * np.pi * fr * t + RNG.uniform(0, 6.28)) * np.exp(-t / dec)
    att = butter(noise(n), 4000, 'high', 2) * env_ad(n, 0.0001, 0.0015) * 0.2
    x = (x + att) * np.minimum(1, t / 0.0008)
    x *= np.minimum(1, (dur - t) / 0.05)
    return x * vel * 0.25


def pad_chord(midis, dur, attack=0.05, release=0.25, cutoff=2600.0, voices=7, detune=0.22, bright_env=True):
    """Supersaw pad chord, stereo. dur = held length (sec); release appended."""
    n = int((dur + release) * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    spread = np.linspace(-1, 1, voices)
    for m in midis:
        f0 = midi_hz(m)
        for v in range(voices):
            cents = spread[v] * detune * 100 * (0.6 + 0.4 * RNG.random())
            sig = polyblep_saw(f0 * 2 ** (cents / 1200), n, RNG.random())
            out += pan(sig, spread[v] * 0.85) / voices
    env = np.clip(t / attack, 0, 1) * np.where(t < dur, 1.0, np.exp(-(t - dur) / (release / 3)))
    fc = cutoff * (1 + (0.6 * np.exp(-t / 0.25) if bright_env else 0))
    out = svf(out, fc, 0.9, 'lp')
    out = butter(out, 140, 'high', 2)
    return out * env[:, None] / max(1, len(midis)) * 1.2


def sub_note(m, dur, release=0.08):
    n = int((dur + release) * SR)
    t = np.arange(n) / SR
    env = np.clip(t / 0.01, 0, 1) * np.where(t < dur, 1.0, np.exp(-(t - dur) / (release / 3)))
    x = sine(midi_hz(m), n) + 0.15 * np.sin(2 * 2 * np.pi * midi_hz(m) * t)
    return np.tanh(1.3 * x) * env


# ---------------------------------------------------------------- FX
def crash(dur=2.6):
    n = int(dur * SR)
    t = np.arange(n) / SR
    metal = sum(np.sin(2 * np.pi * f * t + RNG.uniform(0, 6.28)) for f in RNG.uniform(3000, 11000, 40))
    x = noise(n) + 0.15 * metal
    x = butter(x, 3500, 'high', 2)
    e = env_ad(n, 0.0008, 0.75) * 0.7 + env_ad(n, 0.0005, 0.06) * 0.6
    l = x * e
    r = np.roll(butter(noise(n), 3500, 'high', 2) * e, 0)
    return np.stack([_norm(l), _norm(0.5 * l + 0.5 * r)], axis=1)


def impact(dur=2.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 32 + 55 * np.exp(-t / 0.08)
    boom = np.tanh(2.0 * sine(f, n) * env_ad(n, 0.002, 0.55))
    nz = butter(noise(n), 1800, 'low', 2) * env_ad(n, 0.001, 0.12) * 0.6
    return _norm(boom + nz)


def riser(dur, f0=250.0, f1=9000.0, tone=True):
    """Noise sweep riser + optional rising saw tone, stereo, ends at full level."""
    n = int(dur * SR)
    u = np.arange(n) / n
    fc = f0 * (f1 / f0) ** (u ** 1.6)
    nl = svf(noise(n), fc, 3.0, 'bp')
    nr = svf(noise(n), fc * 1.05, 3.0, 'bp')
    x = np.stack([nl, nr], axis=1)
    if tone:
        ff = midi_hz(nm('C3')) * 2 ** (3 * u ** 1.8)
        s = polyblep_saw(ff, n) + polyblep_saw(ff * 1.006, n)
        s = svf(s, 600 + 6000 * u ** 2, 1.2, 'lp') * 0.35
        x += np.stack([s, s], axis=1)
    amp = (u ** 2.2)
    amp[-int(0.01 * SR):] *= np.linspace(1, 0, int(0.01 * SR))
    return x * amp[:, None] / (np.abs(x).max() + 1e-9)


def flash(dur=1.6):
    """Camera flash 'whine + pop' burst."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    whine = sine(3500 + 4500 * np.clip(t / 0.25, 0, 1), n) * env_ad(n, 0.01, 0.12) * 0.12
    pop = butter(noise(n), 400, 'high', 2) * env_ad(n, 0.0002, 0.09)
    return _norm(pop + whine)


def reverse_crash(dur):
    c = crash(dur + 0.3)[:int(dur * SR)]
    return c[::-1].copy()


def crackle(dur, density=9.0, level=1.0):
    """Vinyl/old-film crackle: sparse random clicks + faint band-limited hiss, stereo."""
    n = int(dur * SR)
    out = np.zeros((n, 2))
    k = RNG.poisson(density * dur)
    pos = RNG.integers(0, max(1, n - 400), k)
    for p in pos:
        L = RNG.integers(20, 160)
        c = noise(L) * np.exp(-np.arange(L) / (L / 4)) * RNG.uniform(0.1, 1.0) ** 2
        out[p:p + L, RNG.integers(0, 2)] += c
        out[p:p + L, :] += c[:, None] * 0.3
    hiss = np.stack([butter(noise(n), [800, 6000], 'band', 2) for _ in range(2)], axis=1) * 0.02
    out = butter(out, 900, 'high', 2) + hiss
    return out / (np.abs(out).max() + 1e-9) * level
