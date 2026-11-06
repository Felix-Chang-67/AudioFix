"""Majority-vote the three chorus transcriptions into one canonical chorus melody (14 bars)."""
import sys, json, numpy as np
mel = json.load(open(sys.argv[1]))
for nt in mel:
    while nt['midi'] < 55: nt['midi'] += 12
    while nt['midi'] > 76: nt['midi'] -= 12
STARTS = [32, 70, 108]
PC = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
def nm(m): return f"{PC[m%12]}{m//12-1}"
grid = []
for st in STARTS:
    g = {}
    for nt in mel:
        rel = nt['slot'] - st*16
        if 0 <= rel < 14*16: g[rel] = (nt['midi'], nt['len'])
    grid.append(g)
voted = []
agree = {3: 0, 2: 0, 1: 0}
for s in range(14*16):
    c = [g[s] for g in grid if s in g]
    if not c: continue
    pitches = [m for m, _ in c]
    best = max(set(pitches), key=pitches.count); n = pitches.count(best)
    # a note survives if >=2 choruses start a note here (same pitch), or chorus 1 has it and nobody disagrees on pitch
    if n >= 2 or (len(c) == 1 and s in grid[0]):
        L = int(np.median([l for m, l in c if m == best]))
        voted.append({'slot': s, 'midi': best, 'len': L, 'votes': n})
        agree[min(n, 3)] += 1
# fix overlaps
for i in range(len(voted)-1):
    voted[i]['len'] = max(1, min(voted[i]['len'], voted[i+1]['slot'] - voted[i]['slot']))
json.dump(voted, open(sys.argv[2], 'w'))
print(f"voted notes {len(voted)}; votes 3:{agree[3]} 2:{agree[2]} 1(chorus1 only):{agree[1]}")
for b in range(14):
    for h in range(2):
        ns = [n for n in voted if b*16 + h*8 <= n['slot'] < b*16 + h*8 + 8]
        print(f"  bar {b+1:2d}{'ab'[h]}: " + "  ".join(f"{n['slot']%16:2d}:{nm(n['midi'])}x{n['len']}({n['votes']})" for n in ns))
