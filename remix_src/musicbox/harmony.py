"""Half-bar chord chart for the whole song (source bars), diatonic triads + power chords."""
import sys, json, numpy as np, soundfile as sf, librosa
stems, bassfile, out = sys.argv[1], sys.argv[2], sys.argv[3]
BPM, PH = 145.01, 0.015; BAR = 240/BPM; BEAT = BAR/4
names = ['bass','drums','other','vocals','guitar','piano']
S = {}
for i, n in enumerate(names):
    x, sr = sf.read(f"{stems}/0{i+1}_shutter_doll_{n}.wav", dtype='float32'); S[n] = x.mean(1)
h = librosa.resample(S['guitar'] + S['other'] + S['piano'] + 0.5*S['bass'], orig_sr=sr, target_sr=11025)
C = librosa.feature.chroma_cqt(y=h, sr=11025, hop_length=256, fmin=librosa.note_to_hz('C2'), n_octaves=5)
ct = librosa.times_like(C, sr=11025, hop_length=256)
bm = np.load(bassfile)  # per-beat bass midi (nan = unvoiced)
PC = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
VOC = {'Cm':[0,3,7],'Eb':[3,7,10],'Fm':[5,8,0],'F':[5,9,0],'Gm':[7,10,2],'G':[7,11,2],'Ab':[8,0,3],'Bb':[10,2,5],
       'C5':[0,7],'F5':[5,0],'G5':[7,2],'Ab5':[8,3],'Bb5':[10,5],'Eb5':[3,10],'D5':[2,9]}
T = {}
for k, pcs in VOC.items():
    v = np.zeros(12); v[pcs] = 1; T[k] = v/np.linalg.norm(v)
chart = {}
for b in range(8, 127):
    for half in (0, 1):
        t0 = PH + b*BAR + half*BAR/2
        sel = (ct >= t0 + 0.03) & (ct < t0 + BAR/2 - 0.03)
        c = C[:, sel].mean(1); c = c/(np.linalg.norm(c)+1e-9)
        sc = {k: float(c @ v) for k, v in T.items()}
        # power chords need a clear win over the triad sharing their root+5th
        best = max(sc, key=sc.get)
        bi = b*4 + half*2
        bpc = None
        for j in (bi, bi+1):
            if j < len(bm) and not np.isnan(bm[j]): bpc = int(round(bm[j] - 0.06)) % 12; break
        chart[f"{b}.{half}"] = {'chord': best, 'score': round(sc[best], 3), 'bass_pc': bpc}
json.dump(chart, open(out, 'w'))
for a, z in [(8, 32), (46, 70), (84, 108)]:
    print(f"--- bars {a}-{z-1}")
    for b in range(a, z, 4):
        print("  " + " | ".join(f"{bb:3d} " + " ".join(f"{chart[f'{bb}.{h}']['chord']:>3}{chart[f'{bb}.{h}']['score']:.2f}/{PC[chart[f'{bb}.{h}']['bass_pc']] if chart[f'{bb}.{h}']['bass_pc'] is not None else '-':<2}" for h in (0,1)) for bb in range(b, b+4)))
