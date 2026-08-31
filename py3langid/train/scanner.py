"""
scanner.py -
Assemble a "feature scanner" using Aho-Corasick string matching.
This takes a list of features (byte sequences) and builds a DFA
that when run on a byte stream can identify, at each position, the
LONGEST feature ending there.

Counting every match (each feature is a suffix of the longest one) makes
Naive Bayes multiply up to five near-identical votes per byte position;
counting only the longest is both more accurate and cheaper. The trie
gives the longest match for free: a state's own feature if it ends there,
otherwise its fail state's, which is the longest proper suffix match.
"""

import array
from collections import defaultdict, deque


def build_scanner(features):
    """Compile a feature list into a DFA transition array (nm_arr, one row
    of 256 next-states per state, next = nm_arr[(state << 8) + byte]) plus
    a state -> longest matched feature index (-1 = no match).

    @param features a list of features (byte sequences)
    @returns (nm_arr, tk_output)
    """
    feat_index = {f: i for i, f in enumerate(features)}

    # trie (Aho-Corasick Algorithm 2), one children dict per state
    children = defaultdict(dict)
    newstate = 0
    ends = {}
    for a in features:
        state = 0
        j = 0
        while j < len(a) and a[j] in children[state]:
            state = children[state][a[j]]
            j += 1
        for p in range(j, len(a)):
            newstate += 1
            children[state][a[p]] = newstate
            state = newstate
        ends[state] = feat_index[a]

    # fail links + DFA fill in one BFS pass: each row starts as a copy of
    # its fail state's row (always shallower, so already complete),
    # then real edges overwrite.
    nstates = newstate + 1
    typecode = 'H' if nstates <= 1 << 16 else 'L'
    nm_arr = array.array(typecode, [0]) * (nstates * 256)
    fail = array.array('L', [0]) * nstates
    tk_output = array.array('l', [-1]) * nstates
    for state, feat in ends.items():
        tk_output[state] = feat
    queue = deque()
    for a, s in children[0].items():
        nm_arr[a] = s
        queue.append(s)  # fail[s] = 0 already
    while queue:
        r = queue.popleft()
        base = r << 8
        fbase = fail[r] << 8
        nm_arr[base:base + 256] = nm_arr[fbase:fbase + 256]
        for a, s in children[r].items():
            fail[s] = nm_arr[base + a]  # = nextmove(fail[r], a), pre-overwrite
            nm_arr[base + a] = s
            if tk_output[s] < 0:
                # no feature ends at s: inherit the longest suffix match
                tk_output[s] = tk_output[fail[s]]
            queue.append(s)
    return nm_arr, tk_output
