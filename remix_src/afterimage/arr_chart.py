"""Chord chart + form for 'Shutter Doll (Afterimage)' — piano & strings rearrangement.

All positions are OUTPUT bars on the 145.01 BPM grid (halftime feel). Chord names are written
in C minor; bars >= KEY_UP_FROM are transposed up a semitone (C# minor) when rendered.
"""
import re

KEY_UP_FROM = 58.5          # vocal pickup into the final chorus is already in the new key

# vocal segments: (src_bar_start, src_bar_end, out_bar_of_src_bar_start_offset, semitones)
#   out_bar = src_bar - offset
VOCAL_SEGMENTS = [
    (16.25, 46.30, 8, 0),    # verse 1 + chorus 1 (ends after the last chorus word)
    (91.95, 104.0, 53, 0),   # bridge, moved to follow chorus 1
    (107.5, 127.07, 49, 1),  # pickup + final chorus + tag, up a semitone
]

SECTIONS = [  # (name, out_bar_start, out_bar_end)
    ('intro', 0, 8), ('verse', 8, 24), ('chorus', 24, 38), ('turn', 38, 39), ('bridge', 39, 51),
    ('strings', 51, 59), ('final', 59, 73), ('tag', 73, 78), ('outro', 78, 83),
]

C = {}
def put(bar, *syms):
    C[bar] = list(syms) if len(syms) == 2 else [syms[0], syms[0]]

# intro (piano solo, chorus hook in augmentation)
for b, s in enumerate(['Fm9', 'Fm9', 'Ebmaj7/G', 'Ebmaj7/G', 'Abmaj7', 'Abmaj7', 'Aø7', 'G7sus4']):
    put(b, s)
# verse 1: the C-chant over moving harmony, then a chromatic lament bass C-B-Bb-A-Ab
put(8, 'Abmaj7'); put(9, 'Abmaj7'); put(10, 'Abmaj7', 'Eb/G'); put(11, 'Gm7'); put(12, 'Fm9', 'G7sus4')
put(13, 'Cm(add9)'); put(14, 'Cm/B'); put(15, 'Cm7/Bb'); put(16, 'Aø7'); put(17, 'Abmaj7')
put(18, 'Fm9', 'Fm6'); put(19, 'G7sus4', 'G7b9'); put(20, 'Abmaj7'); put(21, 'Dbmaj7')
put(22, 'G7b9b13'); put(23, 'G7b9b13')
# chorus: bass climbs F-G-Ab-A-Bb-G-C, then falls Bb-Ab-G-F-Eb-Db
CHORUS = [('Fm9',), ('Ebmaj7/G',), ('Abmaj7',), ('Aø7',), ('Bb6/9',), ('G7sus4',), ('Cm7',),
          ('Cm7/Bb', 'Fm7/Ab'), ('Abmaj7',), ('G7sus4',), ('Gm7/F',), ('Ebmaj7', 'Fm7/Eb'), ('Dbmaj7',), ('Eb/Bb', 'G7')]
for i, s in enumerate(CHORUS):
    put(24 + i, *s)
put(38, 'Cm(add9)')
# bridge: relative-major colour (IV - vi - V - I in Eb), then back towards the dominant
for b, s in zip(range(39, 51), ['Ab(add9)', 'Cm7', 'Bb9sus4', 'Eb6', 'Abmaj9', 'Ebmaj7/G', 'Fm9', 'Dbmaj9',
                                'Abmaj7', 'Fm7', 'Bb7sus4', 'G7sus4']):
    put(b, s)
# string interlude: violins sing the chorus; pivot to Ab7 (= V of C# minor)
for i, s in enumerate(CHORUS[:7]):
    put(51 + i, *s)
put(58, 'Cm7', 'G7#5')   # sounds as G#7#5 after the +1 shift
# final chorus (transposed +1 when rendered)
for i, s in enumerate(CHORUS):
    put(59 + i, *s)
put(72, 'Eb/Bb', 'Bb(add9)')   # final chorus ends on Bb-C in the voice, then straight into the tag
# tag: the C-chant again, now over Ab - Fm - Db - Bb - Cm
for b, s in zip(range(73, 78), ['Abmaj7', 'Fm9', 'Dbmaj7', 'Bb(add9)', 'Cm(add9)']):
    put(b, s)
# outro: piano alone, hook head, Picardy third
put(78, 'Cm(add9)'); put(79, 'Fm9'); put(80, 'Ebmaj7/G'); put(81, 'G7sus4'); put(82, 'C(add9)')

NOTE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
QUAL = {
    '': [0, 4, 7], 'm': [0, 3, 7], 'maj7': [0, 4, 7, 11], 'maj9': [0, 4, 7, 11, 2], 'm7': [0, 3, 7, 10],
    'm9': [0, 3, 7, 10, 2], 'm6': [0, 3, 7, 9], 'm(add9)': [0, 3, 7, 2], 'ø7': [0, 3, 6, 10], '7': [0, 4, 7, 10],
    '7sus4': [0, 5, 7, 10], '9sus4': [0, 5, 7, 10, 2], '7b9': [0, 4, 7, 10, 1], '7b9b13': [0, 4, 10, 1, 8],
    '7#5': [0, 4, 8, 10], '6': [0, 4, 7, 9], '6/9': [0, 4, 7, 9, 2], '(add9)': [0, 4, 7, 2],
}


def pc(name):
    v = NOTE[name[0]]
    for ch in name[1:]:
        v += -1 if ch == 'b' else 1
    return v % 12


def parse(sym, shift=0):
    """-> (root_pc, bass_pc, [pcs])"""
    body, _, slash = sym.partition('/')
    if sym.endswith('6/9'):
        body, slash = sym, ''
    m = re.match(r'^([A-G][b#]?)(.*)$', body)
    root, q = pc(m.group(1)), m.group(2)
    pcs = [(root + i) % 12 for i in QUAL[q]]
    bass = pc(slash) if slash else root
    return (root + shift) % 12, (bass + shift) % 12, [(p + shift) % 12 for p in pcs]


def chord_at(out_bar_float):
    b = int(out_bar_float)
    half = 1 if (out_bar_float - b) >= 0.5 else 0
    sym = C[b][half]
    shift = 1 if out_bar_float >= KEY_UP_FROM else 0
    return sym, parse(sym, shift)
