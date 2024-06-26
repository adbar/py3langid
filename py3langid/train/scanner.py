"""
scanner.py -
Assemble a "feature scanner" using Aho-Corasick string matching.
This takes a list of features (byte sequences) and builds a DFA
that when run on a byte stream can identify how often each of
the features is present in a single pass over the stream.

Marco Lui, January 2013

Copyright 2013 Marco Lui <saffsd@gmail.com>. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of the copyright holder.
"""

import argparse
import array
import os
import pickle
from collections import deque, defaultdict
from .common import read_features

class Scanner(object):
    alphabet = map(chr, range(1<<8))
    """
    Implementation of Aho-Corasick string matching.
    This class should be instantiated with a set of keywords, which
    will then be the only tokens generated by the class's search method,
    """
    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            tk_nextmove, tk_output, feats = pickle.load(f)
        if isinstance(feats, dict):
            # The old scanner format had two identical dictionaries as the last
            # two items in the tuple. This format can still be used by langid.py,
            # but it does not carry the feature list, and so cannot be unpacked
            # back into a Scanner object.
            raise ValueError("old format scanner - please retrain. see code for details.")
        # tk_output is a mapping from state to a list of feature indices.
        # because of the way the scanner class is written, it needs a mapping
        # from state to the feature itself. We rebuild this here.
        tk_output_f = dict( (k,[feats[i] for i in v]) for k,v in tk_output.iteritems() )
        scanner = cls.__new__(cls)
        scanner.__setstate__((tk_nextmove, tk_output_f))
        return scanner

    def __init__(self, keywords):
        self.build(keywords)

    def __call__(self, value):
        return self.search(value)

    def build(self, keywords):
        goto = dict()
        fail = dict()
        output = defaultdict(set)

        # Algorithm 2
        newstate = 0
        for a in keywords:
            state = 0
            j = 0
            while (j < len(a)) and (state, a[j]) in goto:
                state = goto[(state, a[j])]
                j += 1
            for p in range(j, len(a)):
                newstate += 1
                goto[(state, a[p])] = newstate
                #print "(%d, %s) -> %d" % (state, a[p], newstate)
                state = newstate
            output[state].add(a)
        for a in self.alphabet:
            if (0,a) not in goto:
                goto[(0,a)] = 0

        # Algorithm 3
        queue = deque()
        for a in self.alphabet:
            if goto[(0,a)] != 0:
                s = goto[(0,a)]
                queue.append(s)
                fail[s] = 0
        while queue:
            r = queue.popleft()
            for a in self.alphabet:
                if (r,a) in goto:
                    s = goto[(r,a)]
                    queue.append(s)
                    state = fail[r]
                    while (state,a) not in goto:
                        state = fail[state]
                    fail[s] = goto[(state,a)]
                    #print "f(%d) -> %d" % (s, goto[(state,a)]), output[fail[s]]
                    if output[fail[s]]:
                        output[s].update(output[fail[s]])

        # Algorithm 4
        self.nextmove = {}
        for a in self.alphabet:
            self.nextmove[(0,a)] = goto[(0,a)]
            if goto[(0,a)] != 0:
                queue.append(goto[(0,a)])
        while queue:
            r = queue.popleft()
            for a in self.alphabet:
                if (r,a) in goto:
                    s = goto[(r,a)]
                    queue.append(s)
                    self.nextmove[(r,a)] = s
                else:
                    self.nextmove[(r,a)] = self.nextmove[(fail[r],a)]

        # convert the output to tuples, as tuple iteration is faster
        # than set iteration
        self.output = dict((k, tuple(output[k])) for k in output)

        # Next move encoded as a single array. The index of the next state
        # is located at current state * alphabet size  + ord(c).
        # The choice of 'H' array typecode limits us to 64k states.
        def generate_nm_arr(typecode):
            def nextstate_iter():
                # State count starts at 0, so the number of states is the number of i
                # the last state (newstate) + 1
                for state in range(newstate+1):
                    for letter in self.alphabet:
                        yield self.nextmove[(state, letter)]
            return array.array(typecode, nextstate_iter())
        try:
            self.nm_arr = generate_nm_arr('H')
        except OverflowError:
            # Could not fit in an unsigned short array, let's try an unsigned long array.
            self.nm_arr = generate_nm_arr('L')

    def __getstate__(self):
        """
        Compiled nextmove and output.
        """
        return (self.nm_arr, self.output)

    def __setstate__(self, value):
        nm_array, output = value
        self.nm_arr = nm_array
        self.output = output
        self.nextmove = {}
        for i, next_state in enumerate(nm_array):
            state = i / 256
            letter = chr(i % 256)
            self.nextmove[(state, letter)] = next_state

    def search(self, string):
        state = 0
        for letter in map(ord,string):
            state = self.nm_arr[(state << 8) + letter]
            for key in self.output.get(state, []):
                yield key

def build_scanner(features):
    """
    In difference to the Scanner class, this function unwraps a layer of indirection in
    the detection of features. It translates the string output of the scanner's output
    mapping into the index values (positions in the list) of the features in the supplied
    feature set. This is very useful where we are only interested in the relative frequencies
    of features.

    @param features a list of features (byte sequences)
    @returns a compiled scanner model
    """
    feat_index = index(features)

    # Build the actual scanner
    print("building scanner")
    scanner = Scanner(features)
    tk_nextmove, raw_output = scanner.__getstate__()

    # tk_output is the output function of the scanner. It should generate indices into
    # the feature space directly, as this saves a lookup
    tk_output = {}
    for k,v in raw_output.items():
        tk_output[k] = tuple(feat_index[f] for f in v)
    return tk_nextmove, tk_output


def index(seq):
    """
    Build an index for a sequence of items. Assumes
    that the items in the sequence are unique.
    @param seq the sequence to index
    @returns a dictionary from item to position in the sequence
    """
    return dict((k,v) for (v,k) in enumerate(seq))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", metavar="INPUT", help="build a scanner for INPUT. If input is a directory, read INPUT/LDfeats")
    parser.add_argument("-o","--output", help="output scanner to OUTFILE", metavar="OUTFILE")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        input_path = os.path.join(args.input, 'LDfeats')
    else:
        input_path = args.input

    if args.output:
        output_path = args.output
    else:
        output_path = input_path + '.scanner'

    # display paths
    print("input path:", input_path)
    print("output path:", output_path)

    nb_features = read_features(input_path)
    tk_nextmove, tk_output = build_scanner(nb_features)
    scanner = tk_nextmove, tk_output, nb_features

    with open(output_path, 'w') as f:
        pickle.dump(scanner, f)
    print("wrote scanner to {0}".format(output_path))
