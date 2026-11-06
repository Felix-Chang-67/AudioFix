import sys, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
d = sys.argv[1]; N = json.load(open(f'{d}/_notes.json')); sec = json.load(open(f'{d}/_sections.json'))
fig, ax = plt.subplots(figsize=(20, 8))
style = {'piano': ('#1f77b4', 2), 'strings pad': ('#2ca02c', 4), 'tremolo strings': ('#17becf', 4), 'violins': ('#ff7f0e', 6),
         'cellos': ('#8c564b', 6), 'contrabass': ('#7f7f7f', 5), 'pizzicato': ('#bcbd22', 5), 'harp': ('#e377c2', 6), 'timpani': ('#000000', 8)}
for name, notes in N['parts'].items():
    if notes:
        t, p = zip(*notes); ax.scatter(t, p, s=style[name][1], c=style[name][0], label=name, alpha=0.7)
t, p = zip(*N['voice']); ax.scatter(t, p, s=10, c='red', marker='s', label='voice (transcribed)')
for k, (a, b) in sec.items():
    ax.axvline(a, color='k', lw=0.7); ax.text(a + 0.5, 100, k, fontsize=10)
ax.set_ylim(24, 104); ax.set_yticks([24, 36, 48, 60, 72, 84, 96]); ax.set_yticklabels(['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'])
ax.grid(alpha=0.3); ax.legend(loc='upper right', ncol=5, markerscale=3, fontsize=9); ax.set_xlabel('seconds')
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=65)
