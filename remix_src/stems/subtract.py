"""ワンミーツハー (One Meets Her) — stem extraction by subtraction.

usage: python3 -I subtract.py <src_dir> <out_dir>

<src_dir> holds the full mix and the "minus one" versions:
    01 ワンミーツハー.wav                  full mix (reference timeline)
    02 ワンミーツハー [Instrumental].wav   -> vocals  = full - instrumental
    03 ワンミーツハー [non-wowaka].wav     -> wowaka  = full - non-wowaka   (guitar; vocals stay in)
    04 ワンミーツハー [non-shinoda].wav    -> shinoda = full - non-shinoda  (guitar)
    05 ワンミーツハー [non-ygarshy].wav    -> ygarshy = full - non-ygarshy  (bass)
    06 ワンミーツハー [non-yumao].wav      -> yumao   = full - non-yumao    (drums)
plus residual = full - sum(all stems)  (whatever is in no single part, and leftover bleed)

The versions are not sample-aligned (offsets of 10k-26k samples, plus ~5e-8 clock drift) and each
was mastered/limited separately, so a plain sample subtraction does not null. Per version:
  1. alignment: cross-correlation in 5 s windows across the song -> sub-sample lag per window,
     robust linear fit (offset + drift; residual jitter ~0.02 samples), then 64-tap Kaiser-windowed
     sinc resampling onto the full mix's timeline.
  2. EQ match: one complex gain per STFT bin (4096-pt), least squares over the whole song.
  3. limiter match: one real broadband gain per STFT frame, least squares, smoothed over ~0.5 s.
     (Broadband + slow keeps the stem itself intact: on a synthetic mix of known parts it changes
     the stem by about -20 dB, while lowering the real bleed floor by 2-10 dB.)
  4. leakage removal (decorrelate): the guitar/drum differences still carry some vocal and drums;
     regress already-clean stems out per STFT bin (global LS). Everything removed lands in the
     residual, so the six files still sum to the full mix.
Stems are written as 32-bit float WAV, aligned to the full mix, unscaled: several peak above 0 dBFS
and summing all six files returns the full mix.
"""
import sys
import glob
import os
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, stft, istft
from scipy.ndimage import uniform_filter1d

SR = 44100


# ------------------------------------------------------------------ alignment
def measure_lags(y, ref, W=SR * 5, S=SR * 2, step=SR * 5, extra=30000):
    ts, ls, qs = [], [], []
    for t in range(SR * 5, len(ref) - SR * 10, step):
        a = ref[t:t + W].mean(1)
        if np.sqrt((a ** 2).mean()) < 1e-3:
            continue
        s = max(0, t - S - extra)
        b = y[s:s + W + 2 * (S + extra)].mean(1)
        if len(b) < W + 10:
            continue
        c = fftconvolve(b, a[::-1], 'valid')
        k = int(np.argmax(c))
        d = 0.0
        if 0 < k < len(c) - 1:
            y0, y1, y2 = c[k - 1], c[k], c[k + 1]
            d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        q = c[k] / np.sqrt((a ** 2).sum() * (b[k:k + W] ** 2).sum() + 1e-12)
        ts.append(t + W / 2)
        ls.append(s + k + d - t)
        qs.append(q)
    return np.array(ts), np.array(ls), np.array(qs)


def fit_map(ts, ls, qs):
    m = qs > 0.6
    m &= np.abs(ls - np.median(ls[m])) < 3
    for _ in range(3):
        A = np.vstack([np.ones(m.sum()), ts[m]]).T
        coef, *_ = np.linalg.lstsq(A, ls[m], rcond=None)
        res = ls - (coef[0] + coef[1] * ts)
        m &= np.abs(res) < max(0.05, 3 * np.std(res[m]))
    return coef, float(np.std(res[m])), int(m.sum())


def sinc_resample(y, pos, taps=64, beta=10.0, chunk=SR):
    out = np.zeros((len(pos), y.shape[1]))
    half = taps // 2
    ypad = np.pad(y, ((half + 1, half + 1), (0, 0)))
    n = np.arange(-half + 1, half + 1)
    win = np.kaiser(taps, beta)
    for i in range(0, len(pos), chunk):
        p = pos[i:i + chunk]
        ip = np.floor(p).astype(int)
        h = np.sinc(n[None, :] - (p - ip)[:, None]) * win[None, :]
        h /= h.sum(1, keepdims=True)
        valid = (ip >= -half) & (ip < len(y) + half)
        idx = np.clip(ip[:, None] + n[None, :] + half + 1, 0, len(ypad) - 1)
        for c in range(y.shape[1]):
            out[i:i + chunk, c] = (ypad[idx, c] * h).sum(1) * valid
    return out


def align_to(y, ref):
    coef, sd, n = fit_map(*measure_lags(y, ref))
    t = np.arange(len(ref), dtype=float)
    return sinc_resample(y, t + coef[0] + coef[1] * t), coef, sd, n


# ------------------------------------------------------------------ matching + subtraction
NF, HOP = 4096, 1024


def subtract(ref, y, smooth_s=0.5):
    """ref - G(t) * H(f) * y, H complex per bin (global LS), G real broadband per frame (smoothed LS)."""
    kw = dict(nperseg=NF, noverlap=NF - HOP)
    R = [stft(ref[:, c], **kw)[2] for c in range(ref.shape[1])]
    Y = [stft(y[:, c], **kw)[2] for c in range(ref.shape[1])]
    Yh = []
    for Rc, Yc in zip(R, Y):
        H = (Rc * np.conj(Yc)).sum(1) / ((np.abs(Yc) ** 2).sum(1) + 1e-12)
        Yh.append(H[:, None] * Yc)
    sm = max(1, int(round(smooth_s * SR / HOP)))
    num = uniform_filter1d(sum((Rc * np.conj(Yc)).real.sum(0) for Rc, Yc in zip(R, Yh)), sm)
    den = uniform_filter1d(sum((np.abs(Yc) ** 2).sum(0) for Yc in Yh), sm)
    G = np.clip(num / (den + 1e-9 * den.mean() + 1e-15), 0.25, 4.0)
    out = np.zeros_like(ref)
    for c, (Rc, Yc) in enumerate(zip(R, Yh)):
        out[:, c] = istft(Rc - G[None, :] * Yc, **kw)[1][:len(ref)]
    return out


def decorrelate(stems, order):
    """Leakage removal. Each version's master treats the shared parts slightly differently, so e.g.
    'full - non-wowaka' also contains a little vocal and drums. For stem i, the stems order[i]
    (already cleaned) are regressed out with one complex coefficient per STFT bin (global LS, so the
    stem itself is untouched up to its real correlation with those parts)."""
    kw = dict(nperseg=NF, noverlap=NF - HOP)
    N = len(stems[0])
    Z = [np.stack([stft(x[:, c], **kw)[2] for c in range(x.shape[1])]) for x in stems]
    for i, regs in order:
        if not regs:
            continue
        Y = Z[i].copy()
        for c in range(Y.shape[0]):
            X = np.stack([Z[j][c] for j in regs], -1)
            A = np.einsum('ftk,ftl->fkl', np.conj(X), X) + 1e-9 * np.eye(len(regs))
            b = np.einsum('ftk,ft->fk', np.conj(X), Z[i][c])
            K = np.linalg.solve(A, b[..., None])[..., 0]
            Y[c] = Z[i][c] - np.einsum('ftk,fk->ft', X, K)
        Z[i] = Y
    return [np.stack([istft(z[c], **kw)[1][:N] for c in range(z.shape[0])], 1) for z in Z]


STEMS = [  # (filename tag, output name)
    ('[Instrumental]', '01_vocals'),
    ('[non-wowaka]', '02_wowaka_guitar'),
    ('[non-shinoda]', '03_shinoda_guitar'),
    ('[non-ygarshy]', '04_ygarshy_bass'),
    ('[non-yumao]', '05_yumao_drums'),
]


def main(src, out):
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src, '*.wav')))
    full_path = [f for f in files if '[' not in os.path.basename(f)][0]
    ref, sr = sf.read(full_path, dtype='float64')
    assert sr == SR
    raw = []
    for tag, name in STEMS:
        path = [f for f in files if tag in os.path.basename(f)][0]
        y, sr = sf.read(path, dtype='float64')
        assert sr == SR
        ya, coef, sd, n = align_to(y, ref)
        raw.append(subtract(ref, ya))
        print(f'{name:20s} offset {coef[0]:+.2f} smp  drift {coef[1]:+.2e}  jitter {sd:.3f} smp ({n} windows)')
    # 0 vocals, 1 wowaka, 2 shinoda, 3 bass, 4 drums. Vocals (full - instrumental) are the cleanest
    # reference; drums/bass are cleaned against them, the guitars against vocals+bass+drums.
    stems = decorrelate(raw, [(4, [0]), (3, [0]), (1, [0, 3, 4]), (2, [0, 3, 4])])
    stems.append(ref - sum(stems))  # residual: in no single part (e.g. backing vocals) + leftover bleed
    names = [n for _, n in STEMS] + ['06_residual']
    for name, r, s in zip(names, raw + [None], stems):
        sf.write(os.path.join(out, f'One_Meets_Her_{name}.wav'), s.astype(np.float32), SR, subtype='FLOAT')
        rms = 10 * np.log10((s ** 2).mean())
        note = '' if r is None else f'  (leak removed {10 * np.log10((r ** 2).mean()) - rms:.1f} dB)'
        print(f'{name:20s} rms {rms:6.1f} dBFS  peak {np.abs(s).max():.3f}{note}')
    err = np.abs(sum(stems) - ref).max()
    print(f'sum of all stems vs full mix: max abs error {err:.2e}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
