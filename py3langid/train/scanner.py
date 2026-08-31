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
    """Compile a feature list into the DFA the model ships and the runtime
    walks: the distinct 256-entry transition rows, a state -> row index
    (next = rows[(row_index[state] << 8) + byte]), and a state -> longest
    matched feature index (-1 = no match).

    A state with no edges of its own never overwrites the row it inherits
    from its fail state, so it points at that row instead of copying it.
    That is every duplicate row there is (38,270 of 104,518 states for the
    shipped model), so the flat one-row-per-state table is never allocated.

    @param features a list of features (byte sequences)
    @returns (rows, row_index, tk_output)
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

    # fail links + DFA fill in one BFS pass, allocating a row only for states
    # that overwrite something in it (the fail state is always shallower, so
    # its row is already complete)
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
                # no feature ends at s: inherit the longest suffix match
                tk_output[s] = tk_output[fail[s]]
            queue.append(s)
    return rows, row_index, tk_output
