import sys, json
sys.path.insert(0, sys.argv[1])
from arr_chart import VOCAL_SEGMENTS, chord_at
mel = json.load(open(sys.argv[2]))
for nt in mel:
    while nt['midi'] < 55: nt['midi'] += 12
    while nt['midi'] > 76: nt['midi'] -= 12
PC = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
tot = held = in_chord = 0; issues = []
for a, b, off, semi in VOCAL_SEGMENTS:
    for nt in mel:
        sb = nt['slot'] / 16
        if not (a <= sb < b): continue
        tot += 1
        m = (nt['midi'] + semi) % 12
        # check every half-bar the note sounds in
        ob0 = sb - off; ob1 = (nt['slot'] + nt['len']) / 16 - off
        t = ob0
        while t < ob1 - 1e-9:
            sym, (root, bass, pcs) = chord_at(t)
            dur_here = min(ob1, int(t * 2 + 1) / 2) - t
            if nt['len'] >= 2:
                held += 1; in_chord += m in pcs
                d = min(min((m - p) % 12, (p - m) % 12) for p in pcs)
                if m not in pcs and d == 1 and dur_here * 16 >= 2:
                    issues.append((round(t, 2), PC[m], sym, nt['len'], round(dur_here * 16, 1)))
            t = int(t * 2 + 1) / 2
print(f"melody notes {tot}; held-note segments {held}, on chord tones {in_chord} ({in_chord/held:.0%})")
print(f"held notes a semitone from a chord tone: {len(issues)}")
for i in issues: print(f"  out bar {i[0]:6.2f}: melody {i[1]:2s} over {i[2]:9s} (note len {i[3]}, {i[4]} slots in this half-bar)")
