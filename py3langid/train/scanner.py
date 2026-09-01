"""Aho-Corasick DFA: longest-match feature scanner."""

import array
from collections import defaultdict, deque


def build_scanner(features):
    """Compile features into a DFA. Returns (rows, row_index, tk_output)."""
    feat_index = {f: i for i, f in enumerate(features)}

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

    # fail links + DFA fill (row allocated only when a state has its own edges)
    nstates = newstate + 1
    typecode = 'H' if nstates <= 1 << 16 else 'L'
    rows = array.array(typecode, [0]) * 256  # state 0's row
    row_index = array.array('L', [0]) * nstates
    fail = array.array('L', [0]) * nstates
    tk_output = array.array('l', [-1]) * nstates
    for state, feat in ends.items():
        tk_output[state] = feat
    queue = deque()
    for a, s in children[0].items():
        rows[a] = s
        queue.append(s)  # fail[s] = 0 already
    while queue:
        r = queue.popleft()
        fbase = row_index[fail[r]] << 8
        edges = children[r]
        if edges:
            row_index[r] = len(rows) >> 8
            rows.extend(rows[fbase:fbase + 256])
        else:
            row_index[r] = fbase >> 8
        base = row_index[r] << 8
        for a, s in edges.items():
            fail[s] = rows[fbase + a]  # = nextmove(fail[r], a), pre-overwrite
            rows[base + a] = s
            if tk_output[s] < 0:
                tk_output[s] = tk_output[fail[s]]
            queue.append(s)
    return rows, row_index, tk_output
