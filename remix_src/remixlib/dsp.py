"""Low-level DSP helpers for the remix (all arrays: float32/float64, stereo = shape (n, 2))."""
import numpy as np
from numba import njit
import scipy.signal as ss
import pedalboard as pb

SR = 44100
RNG = np.random.default_rng(1234)


def db(x):
    return 10 ** (x / 20.0)


def to_stereo(x):
    return np.stack([x, x], axis=1) if x.ndim == 1 else x


def pan(x, p):
    """Equal-power pan of a mono signal, p in [-1, 1]."""
    a = (p + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)], axis=1)


def add_at(buf, sig, t, gain=1.0):
    """Mix `sig` (mono or stereo) into stereo `buf` starting at time t (seconds)."""
    sig = to_stereo(sig)
    i = int(round(t * SR))
    if i >= len(buf):
        return
    j0 = max(0, -i)
    i = max(0, i)
    n = min(len(sig) - j0, len(buf) - i)
    if n > 0:
        buf[i:i + n] += sig[j0:j0 + n] * gain


@njit(cache=True)
def _svf(x, fc, q, mode, sr):
    y = np.zeros_like(x)
    ic1 = 0.0
    ic2 = 0.0
    k = 1.0 / q
    for n in range(x.shape[0]):
        f = fc[n]
        if f > sr * 0.45:
            f = sr * 0.45
        if f < 10.0:
            f = 10.0
        g = np.tan(np.pi * f / sr)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v3 = x[n] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        if mode == 0:
            y[n] = v2
        elif mode == 1:
            y[n] = v1
        else:
            y[n] = x[n] - k * v1 - v2
    return y


def svf(x, fc, q=0.707, mode='lp'):
    """Time-varying state-variable filter. fc: scalar or per-sample array."""
    m = {'lp': 0, 'bp': 1, 'hp': 2}[mode]
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        return np.stack([svf(x[:, c], fc, q, mode) for c in range(x.shape[1])], axis=1)
    fcv = np.broadcast_to(np.asarray(fc, dtype=np.float64), x.shape).copy()
    return _svf(x, fcv, float(q), m, float(SR))


def butter(x, f, kind='low', order=4):
    sos = ss.butter(order, f, btype=kind, fs=SR, output='sos')
    return ss.sosfilt(sos, x, axis=0)


def polyblep_saw(freq, n, phase0=0.0):
    """Band-limited sawtooth; freq scalar or per-sample array."""
    f = np.broadcast_to(np.asarray(freq, dtype=np.float64), (n,))
    dt = f / SR
    ph = (phase0 + np.cumsum(dt) - dt[0]) % 1.0
    y = 2.0 * ph - 1.0
    m = ph < dt
    xx = ph[m] / dt[m]
    y[m] -= xx + xx - xx * xx - 1.0
    m = ph > 1.0 - dt
    xx = (ph[m] - 1.0) / dt[m]
    y[m] -= xx * xx + xx + xx + 1.0
    return y


def sine(freq, n, phase0=0.0):
    f = np.broadcast_to(np.asarray(freq, dtype=np.float64), (n,))
    return np.sin(2 * np.pi * (np.cumsum(f) / SR) + phase0)


def noise(n):
    return RNG.standard_normal(n)


def env_ad(n, attack, decay, curve=1.0):
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-5), 0, 1)
    d = np.exp(-np.maximum(t - attack, 0) / decay) ** curve
    return a * d


def board(x, plugins):
    """Run a pedalboard chain over a stereo (n,2) array."""
    x = to_stereo(np.asarray(x, dtype=np.float32))
    y = pb.Pedalboard(plugins)(x.T.copy(), SR)
    return y.T.astype(np.float64)


def reverb(x, room=0.8, damp=0.5, width=1.0, predelay=0.0):
    x = to_stereo(x)
    if predelay > 0:
        d = int(predelay * SR)
        x = np.concatenate([np.zeros((d, 2)), x])[:len(x)]
    y = board(x, [pb.Reverb(room_size=room, damping=damp, wet_level=1.0, dry_level=0.0, width=width)])
    # Freeverb's wet output runs ~8-12 dB hot on steady material (measured); normalise to unity
    return y * db(-(8.2 + (room - 0.6) * 11.5))


def delay(x, seconds, feedback=0.4, hp=300, lp=5000, pingpong=True):
    """Feedback delay (wet only), filtered each repeat, optional ping-pong."""
    x = to_stereo(x)
    d = int(round(seconds * SR))
    out = np.zeros_like(x)
    tap = x.mean(axis=1)
    g = 1.0
    side = 0
    for _ in range(12):
        tap = butter(tap, hp, 'high', 2)
        tap = butter(tap, lp, 'low', 2)
        tap = np.concatenate([np.zeros(d), tap])[:len(x)]
        if pingpong:
            out[:, side] += tap * g
            side = 1 - side
        else:
            out += tap[:, None] * g
        g *= feedback
        if g < 0.01:
            break
    return out


def compress(x, thresh=-18, ratio=3, attack=5, release=120, makeup=0):
    return board(x, [pb.Compressor(threshold_db=thresh, ratio=ratio, attack_ms=attack, release_ms=release),
                     pb.Gain(gain_db=makeup)])


def tape_stop(x, start, dur, curve=2.0):
    """Slow playback from speed 1 to 0 over `dur` seconds starting at `start`. Silence afterwards."""
    x = to_stereo(x).copy()
    i0 = int(start * SR)
    n = int(dur * SR)
    u = np.arange(n) / n
    speed = (1 - u) ** curve
    pos = i0 + np.cumsum(speed)
    idx = np.arange(len(x))
    seg = np.stack([np.interp(pos, idx, x[:, c]) for c in range(2)], axis=1)
    fade = np.ones(n)
    fade[-int(0.03 * SR):] = np.linspace(1, 0, int(0.03 * SR))
    x[i0:i0 + n] = seg * fade[:, None]
    x[i0 + n:] = 0
    return x


def automation(points, n, t0=0.0, unit=1.0):
    """Piecewise-linear automation. points: [(time_in_units, value), ...]; returns per-sample array."""
    pts = sorted(points)
    ts = np.array([p[0] for p in pts]) * unit + t0
    vs = np.array([p[1] for p in pts], dtype=np.float64)
    t = np.arange(n) / SR
    return np.interp(t, ts, vs)


def exp_automation(points, n, t0=0.0, unit=1.0):
    """Like automation() but interpolates log-values (for filter cutoffs)."""
    pts = [(a, np.log(b)) for a, b in points]
    return np.exp(automation(pts, n, t0, unit))


def true_peak_db(x):
    up = ss.resample_poly(x, 4, 1, axis=0)
    return 20 * np.log10(np.abs(up).max() + 1e-12)


@njit(cache=True)
def _release(g, coef):
    out = np.empty_like(g)
    cur = 1.0
    for i in range(g.shape[0]):
        if g[i] < cur:
            cur = g[i]
        else:
            cur = cur + (g[i] - cur) * coef
        out[i] = cur
    return out


def limiter(x, ceiling_db=-1.0, lookahead=0.0025, release=0.09):
    """Look-ahead brickwall limiter on 4x-oversampled peaks (offline, zero latency)."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    c = db(ceiling_db)
    up = ss.resample_poly(x, 4, 1, axis=0)
    pk = np.abs(up).max(axis=1)
    pk = pk[:len(x) * 4].reshape(-1, 4).max(axis=1)
    g = np.minimum(1.0, c / (pk + 1e-12))
    L = max(1, int(lookahead * SR))
    g = minimum_filter1d(g, size=2 * L + 1)
    g = _release(g, 1 - np.exp(-1 / (release * SR)))
    g = uniform_filter1d(g, size=L)
    return x * g[:, None], g
