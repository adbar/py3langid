#!/usr/bin/env python3
"""Language identification (fork of langid.py by Marco Lui)."""

import logging
import math
import re
import sys
import unicodedata
from collections import Counter
from operator import itemgetter
from pathlib import Path

import numpy as np

from .modelio import load_model

LOGGER = logging.getLogger(__name__)

IDENTIFIER = None
MODEL_DIR = Path(__file__).parent
MODEL_FILE = MODEL_DIR / 'data/model.npz.xz'
RAW_FLOOR = float(np.finfo(np.float32).min)  # finite floor for featureless input


def decode_trimmed(data):
    """Decode UTF-8, trimming ≤3 partial trailing bytes; None if undecodable.
    Shared train/inference contract (normalize)."""
    for trim in range(4):
        chunk = data[:len(data) - trim] if trim else data
        try:
            return chunk.decode('utf8')
        except UnicodeDecodeError as e:
            if e.start < len(data) - 3:  # not fixable by trimming the tail
                return None
    return None


def normalize(text):
    """(lowercased NFC UTF-8 bytes, str) of *text*; undecodable bytes pass through.
    Shared train/inference contract (train.common.read_doc)."""
    if isinstance(text, bytes):
        decoded = decode_trimmed(text)
        if decoded is None:
            return text, text.decode('utf8', errors='replace')
        text = decoded
    text = unicodedata.normalize('NFC', text.lower())
    return text.encode('utf8', errors='surrogatepass'), text


_CJK = r"\u2E80-\u9FFF\uF900-\uFAFF\uFF00-\uFFEF"  # one token per character
CJK_RE = re.compile(f"[{_CJK}]")
TOKEN_RE = re.compile(f"[{_CJK}]|[^\\W\\d_{_CJK}]+")


def visit_counts(nm, rowbase, out, text):
    """DFA-walk feature counts over bytes.
    Shared by inference (_raw_score) and training (train.stages)."""
    state, indexes = 0, []
    append = indexes.append
    for letter in text:
        state = nm[rowbase[state] + letter]
        f = out[state]
        if f >= 0:
            append(f)
    return Counter(indexes)


def _load_identifier(model_path=None, norm_probs=False, langs=None):
    identifier = LanguageIdentifier.from_modelpath(model_path or MODEL_FILE, norm_probs=norm_probs)
    if langs:
        identifier.set_languages(langs)
    return identifier


def _get_identifier(model_path=None, norm_probs=False, langs=None):
    """Module identifier, loaded on first use (also the batch pool initializer:
    forked workers inherit the parent's, spawned ones load their own)."""
    global IDENTIFIER
    if IDENTIFIER is None:
        LOGGER.debug('initializing identifier')
        IDENTIFIER = _load_identifier(model_path, norm_probs, langs)
    return IDENTIFIER


def set_languages(langs=None):
    return _get_identifier().set_languages(langs)


def classify(instance):
    return _get_identifier().classify(instance)


def rank(instance):
    return _get_identifier().rank(instance)


def _process_file(path, dist=False):
    with open(path, 'rb') as f:
        text = f.read()
    return path, (rank(text) if dist else classify(text))


class LanguageIdentifier:
    __slots__ = ['_all_labels', '_dupes', '_first', '_model', '_norm_probs',
                 '_rowbase', '_sel', '_words', 'labels', 'min_confidence']

    @classmethod
    def from_modelpath(cls, path, *args, **kwargs):
        return cls(load_model(path), *args, **kwargs)

    @classmethod
    def from_model_file(cls, model_file, *args, **kwargs):
        """0.4.0 API: relative to the package directory."""
        return cls.from_modelpath(MODEL_DIR / model_file, *args, **kwargs)

    def __init__(self, model, norm_probs=False, min_confidence=None):
        if min_confidence is not None and not norm_probs:
            raise ValueError("min_confidence requires norm_probs=True")
        self.min_confidence = min_confidence
        self._model = model
        w = model.words
        index = {tok: i for i, tok in enumerate(w.vocab.decode('utf8').split('\n'))}
        self._words = (index, w.indptr, w.cols, w.vals)
        self._rowbase = [r << 8 for r in model.row]  # pre-shifted row offsets
        self._norm_probs = norm_probs
        groups = {}  # label -> columns
        for i, c in enumerate(model.classes):
            groups.setdefault(c, []).append(i)
        self._all_labels = list(groups)
        self._first = np.array([g[0] for g in groups.values()])
        self._dupes = [(k, j) for k, g in enumerate(groups.values()) for j in g[1:]]
        self.set_languages(None)

    def set_languages(self, langs=None):
        """Restrict classification to *langs* (ISO 639 codes), or reset to all."""
        LOGGER.debug("restricting languages to: %s", langs)
        if langs is None:
            self.labels, self._sel = self._all_labels, None
        else:
            if not langs:
                raise ValueError("Empty language selection")
            wanted = set(langs)
            unknown = wanted - set(self._all_labels)
            if unknown:
                raise ValueError(f"Unknown language code(s): {unknown}")
            self._sel = np.array([i for i, c in enumerate(self._all_labels) if c in wanted])
            self.labels = [self._all_labels[i] for i in self._sel]

    def _raw_score(self, text):
        """NB log-posterior from the DFA walk's sparse feature counts, None if featureless."""
        visits = visit_counts(self._model.nextmove, self._rowbase, self._model.output, text)
        if not visits:
            return None
        idx = np.fromiter(visits.keys(), dtype=np.intp, count=len(visits))
        counts = np.fromiter(visits.values(), dtype=np.float32, count=len(visits))
        return np.log1p(counts) @ self._model.ptc[idx] + self._model.pc

    def _word_credit(self, decoded):
        """Summed table credits over the distinct known tokens, per column."""
        index, indptr, cols, vals = self._words
        rows = {index[w] for w in TOKEN_RE.findall(decoded) if w in index}
        if not rows:
            return None
        spans = [slice(indptr[r], indptr[r + 1]) for r in rows]
        return np.bincount(np.concatenate([cols[s] for s in spans]),
                           weights=np.concatenate([vals[s] for s in spans]),
                           minlength=len(self._model.pc))

    def _decide(self, text):
        """Scores in self.labels order, probabilities under norm_probs."""
        data, decoded = normalize(text)
        text = b' ' + data + b' '  # padding lets boundary n-grams fire on short input
        scores = self._raw_score(text)
        credit = self._word_credit(decoded)
        if scores is None:  # featureless: uniform under norm_probs, RAW_FLOOR otherwise
            if self._norm_probs and credit is None:
                return np.full(len(self.labels), 1 / len(self.labels), dtype=np.float32)
            scores = np.full(len(self._model.pc), 0.0 if self._norm_probs else RAW_FLOOR,
                             dtype=np.float32)
        if credit is not None:
            scores += credit
        if self._norm_probs:
            scores *= 1.0 / math.sqrt(len(text))  # T = sqrt(bytes)
        out = scores[self._first]  # aliased label: best column, or summed as probability
        for k, j in self._dupes:
            out[k] = np.logaddexp(out[k], scores[j]) if self._norm_probs else max(out[k], scores[j])
        if self._sel is not None:
            out = out[self._sel]
        if self._norm_probs:
            np.exp(out - out.max(), out=out)
            out /= out.sum()
        return out

    def classify(self, text):
        """Return *(language, confidence)* for *text* (str or UTF-8 bytes)."""
        scores = self._decide(text)
        i = int(scores.argmax())
        conf = float(scores[i])
        if self.min_confidence is not None and conf < self.min_confidence:
            return 'und', conf
        return self.labels[i], conf

    def rank(self, text):
        """All languages by likelihood, best first."""
        return sorted(zip(self.labels, self._decide(text).tolist()), key=itemgetter(1), reverse=True)


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--serve', action='store_true', help='launch web service')
    parser.add_argument('--host', help='host/ip to bind to')
    parser.add_argument('--port', default=9008, type=int, help='port to listen on')
    parser.add_argument('-v', action='count', dest='verbosity', help='increase verbosity (repeat for greater effect)')
    parser.add_argument('-m', dest='model', help='load model from file')
    parser.add_argument('-l', '--langs', help='comma-separated set of target ISO639 language codes (e.g en,de)')
    parser.add_argument('-b', '--batch', action='store_true', help='read file paths from stdin and classify in parallel')
    parser.add_argument('-d', '--dist', action='store_true', help='show full distribution over languages')
    parser.add_argument('--line', action='store_true', help='process pipes line-by-line rather than as a document')
    parser.add_argument('-n', '--normalize', action='store_true', help='normalize confidence scores to probability values')
    return parser


def _run_serve(host, port):
    """Serve the WSGI app until interrupted."""
    import socket
    from wsgiref.simple_server import make_server

    from .server import application

    hostname = host or socket.gethostbyname(socket.gethostname())
    print(f"Listening on {hostname}:{port}")
    print("Press Ctrl+C to exit")
    httpd = make_server(hostname, port, application)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def _stdin_paths():
    """Existing file paths, one per stdin line."""
    for line in sys.stdin:
        path = line.strip()
        if path and Path(path).is_file():
            yield path


def _run_batch(identifier, options):
    """Classify the file paths on stdin in parallel, as CSV on stdout."""
    import csv
    import multiprocessing as mp
    from functools import partial

    writer = csv.writer(sys.stdout, lineterminator='\n')
    ctx = mp.get_context('fork') if sys.platform == 'darwin' else mp
    with ctx.Pool(processes=mp.cpu_count(),
                  initializer=_get_identifier,
                  initargs=(options.model, options.normalize, identifier.labels)) as pool:
        if options.dist:
            header = identifier.labels
            writer.writerow(['path', 'language'] + header)
            for path, ranking in pool.imap_unordered(partial(_process_file, dist=True), _stdin_paths()):
                scores = dict(ranking)
                writer.writerow([path, ranking[0][0]] + [scores[c] for c in header])
        else:
            for path, (lang, conf) in pool.imap_unordered(_process_file, _stdin_paths()):
                writer.writerow((path, lang, conf))


def _run_stdin(process, line_mode):
    """Interactive prompt on a tty, otherwise classify piped input."""
    if sys.stdin.isatty():
        while True:
            try:
                print(">>>", end=' ')
                text = input()
            except (KeyboardInterrupt, EOFError):
                break
            print(process(text))
    elif line_mode:
        for line in sys.stdin:
            print(process(line))
    else:
        print(process(sys.stdin.read()))


def main(argv=None):
    global IDENTIFIER

    parser = _build_parser()
    options = parser.parse_args(argv)

    if options.verbosity:
        logging.basicConfig(level=max((5 - options.verbosity) * 10, 0))
    else:
        logging.basicConfig()

    if options.batch and options.serve:
        parser.error("cannot specify both batch and serve at the same time")

    langs = options.langs.split(",") if options.langs else None
    IDENTIFIER = _load_identifier(options.model, options.normalize, langs)

    if options.serve:
        _run_serve(options.host, options.port)
    elif options.batch:
        _run_batch(IDENTIFIER, options)
    else:
        _run_stdin(IDENTIFIER.rank if options.dist else IDENTIFIER.classify, options.line)


if __name__ == "__main__":
    main()
