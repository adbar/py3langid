#!/usr/bin/env python3
"""Language identification (fork of langid.py by Marco Lui)."""

import logging
import math
import re
import sys
import unicodedata
from itertools import pairwise
from operator import itemgetter
from pathlib import Path

import numpy as np

from .dfa import build_dfa, visit_counts
from .modelio import load_model

LOGGER = logging.getLogger(__name__)

IDENTIFIER = None
MODEL_DIR = Path(__file__).parent
MODEL_FILE = MODEL_DIR / 'data/model.npz.xz'
RAW_FLOOR = float(np.finfo(np.float32).min)  # finite floor for featureless input


def normalize(text):
    """(lowercased NFC UTF-8 bytes, str) of *text*; bytes lose a partial trailing codepoint,
    undecodable bytes pass through. Shared train/inference contract (train.common.read_doc)."""
    if isinstance(text, bytes):
        try:
            text = text.decode('utf8')
        except UnicodeDecodeError as e:
            if e.start < len(text) - 3:  # not fixable by trimming the tail
                text = unicodedata.normalize('NFC', text.decode('utf8', errors='surrogateescape').lower())
                return text.encode('utf8', errors='surrogateescape'), text
            text = text[:e.start].decode('utf8')
    text = unicodedata.normalize('NFC', text.lower())
    return text.encode('utf8', errors='surrogatepass'), text


CJK = r"\u2E80-\u9FFF\uF900-\uFAFF\uFF00-\uFFEF"  # one token per character
# BMP combining marks (Unicode 16, fixed across Python versions) continue a word
_MARKS = (
    r"\u0300-\u036F\u0483-\u0489\u0591-\u05BD\u05BF\u05C1-\u05C2\u05C4-\u05C5\u05C7"
    r"\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7-\u06E8\u06EA-\u06ED"
    r"\u0711\u0730-\u074A\u07A6-\u07B0\u07EB-\u07F3\u07FD\u0816-\u0819\u081B-\u0823"
    r"\u0825-\u0827\u0829-\u082D\u0859-\u085B\u0897-\u089F\u08CA-\u08E1\u08E3-\u0903"
    r"\u093A-\u093C\u093E-\u094F\u0951-\u0957\u0962-\u0963\u0981-\u0983\u09BC\u09BE-\u09C4"
    r"\u09C7-\u09C8\u09CB-\u09CD\u09D7\u09E2-\u09E3\u09FE\u0A01-\u0A03\u0A3C\u0A3E-\u0A42"
    r"\u0A47-\u0A48\u0A4B-\u0A4D\u0A51\u0A70-\u0A71\u0A75\u0A81-\u0A83\u0ABC\u0ABE-\u0AC5"
    r"\u0AC7-\u0AC9\u0ACB-\u0ACD\u0AE2-\u0AE3\u0AFA-\u0AFF\u0B01-\u0B03\u0B3C\u0B3E-\u0B44"
    r"\u0B47-\u0B48\u0B4B-\u0B4D\u0B55-\u0B57\u0B62-\u0B63\u0B82\u0BBE-\u0BC2\u0BC6-\u0BC8"
    r"\u0BCA-\u0BCD\u0BD7\u0C00-\u0C04\u0C3C\u0C3E-\u0C44\u0C46-\u0C48\u0C4A-\u0C4D"
    r"\u0C55-\u0C56\u0C62-\u0C63\u0C81-\u0C83\u0CBC\u0CBE-\u0CC4\u0CC6-\u0CC8\u0CCA-\u0CCD"
    r"\u0CD5-\u0CD6\u0CE2-\u0CE3\u0CF3\u0D00-\u0D03\u0D3B-\u0D3C\u0D3E-\u0D44\u0D46-\u0D48"
    r"\u0D4A-\u0D4D\u0D57\u0D62-\u0D63\u0D81-\u0D83\u0DCA\u0DCF-\u0DD4\u0DD6\u0DD8-\u0DDF"
    r"\u0DF2-\u0DF3\u0E31\u0E34-\u0E3A\u0E47-\u0E4E\u0EB1\u0EB4-\u0EBC\u0EC8-\u0ECE"
    r"\u0F18-\u0F19\u0F35\u0F37\u0F39\u0F3E-\u0F3F\u0F71-\u0F84\u0F86-\u0F87\u0F8D-\u0F97"
    r"\u0F99-\u0FBC\u0FC6\u102B-\u103E\u1056-\u1059\u105E-\u1060\u1062-\u1064\u1067-\u106D"
    r"\u1071-\u1074\u1082-\u108D\u108F\u109A-\u109D\u135D-\u135F\u1712-\u1715\u1732-\u1734"
    r"\u1752-\u1753\u1772-\u1773\u17B4-\u17D3\u17DD\u180B-\u180D\u180F\u1885-\u1886\u18A9"
    r"\u1920-\u192B\u1930-\u193B\u1A17-\u1A1B\u1A55-\u1A5E\u1A60-\u1A7C\u1A7F\u1AB0-\u1ACE"
    r"\u1B00-\u1B04\u1B34-\u1B44\u1B6B-\u1B73\u1B80-\u1B82\u1BA1-\u1BAD\u1BE6-\u1BF3"
    r"\u1C24-\u1C37\u1CD0-\u1CD2\u1CD4-\u1CE8\u1CED\u1CF4\u1CF7-\u1CF9\u1DC0-\u1DFF"
    r"\u20D0-\u20F0\u2CEF-\u2CF1\u2D7F\u2DE0-\u2DFF\u302A-\u302F\u3099-\u309A\uA66F-\uA672"
    r"\uA674-\uA67D\uA69E-\uA69F\uA6F0-\uA6F1\uA802\uA806\uA80B\uA823-\uA827\uA82C"
    r"\uA880-\uA881\uA8B4-\uA8C5\uA8E0-\uA8F1\uA8FF\uA926-\uA92D\uA947-\uA953\uA980-\uA983"
    r"\uA9B3-\uA9C0\uA9E5\uAA29-\uAA36\uAA43\uAA4C-\uAA4D\uAA7B-\uAA7D\uAAB0\uAAB2-\uAAB4"
    r"\uAAB7-\uAAB8\uAABE-\uAABF\uAAC1\uAAEB-\uAAEF\uAAF5-\uAAF6\uABE3-\uABEA\uABEC-\uABED"
    r"\uFB1E\uFE00-\uFE0F\uFE20-\uFE2F"
)
TOKEN_RE = re.compile(f"[{CJK}]|[^\\W\\d_{CJK}](?:[^\\W\\d_{CJK}]|[{_MARKS}])*")
HAN = r"\u3400-\u9FFF\uF900-\uFAFF"
HAN_PAIR_RE = re.compile(f"(?=([{HAN}]{{2}}))")  # overlapping Han pairs
HAN_RE = re.compile(f"[{HAN}]")
LATIN = r"a-z\u00C0-\u024F\u1E00-\u1EFF"  # input is lowercased
LATIN_RE = re.compile(f"[{LATIN}]")
LATIN_RUN_RE = re.compile(f"[{LATIN}0-9]+")
LATIN_STRIP_RATIO = 3  # Latin runs dropped up to this many letters per Han char


def log_probs(counts, n_features, n_classes):
    """Add-one smoothed NB log-probabilities (features, classes) as float16, one class at a time."""
    bounds = np.searchsorted(counts.index, np.arange(n_classes + 1) * n_features).tolist()
    spans = list(pairwise(bounds))
    log_total = np.log(n_features + np.array([counts.values[a:b].sum() for a, b in spans], dtype=np.int64))
    ptc = np.empty((n_features, n_classes), dtype=np.float16)
    ptc[:] = (-log_total).astype(np.float16)  # log(1 + 0) - log_total
    for c, (a, b) in enumerate(spans):
        ptc[counts.index[a:b] - c * n_features, c] = np.log(1.0 + counts.values[a:b]) - log_total[c]
    return ptc


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
    __slots__ = ['_dfa', '_dupes', '_first', '_groups', '_model', '_norm_probs',
                 '_ptc', '_words', 'labels', 'min_confidence']

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
        self._ptc = log_probs(model.counts, len(model.features), len(model.pc))
        self._dfa = build_dfa(model.features)
        self._model = model._replace(counts=None, features=None)  # ptc and the DFA replace them
        w = model.words
        index = {tok: i for i, tok in enumerate(w.vocab.decode('utf8').split('\n'))}
        self._words = (index, w.indptr, w.cols, w.vals)
        self._norm_probs = norm_probs
        self._groups = {}  # label -> columns
        for i, c in enumerate(model.classes):
            self._groups.setdefault(c, []).append(i)
        self.set_languages(None)

    def set_languages(self, langs=None):
        """Restrict classification to *langs* (ISO 639 codes), or reset to all."""
        LOGGER.debug("restricting languages to: %s", langs)
        wanted = self._groups.keys() if langs is None else set(langs)
        if not wanted:
            raise ValueError("Empty language selection")
        unknown = wanted - self._groups.keys()
        if unknown:
            raise ValueError(f"Unknown language code(s): {unknown}")
        self.labels = [c for c in self._groups if c in wanted]
        cols = [self._groups[c] for c in self.labels]
        self._first = np.array([g[0] for g in cols])
        self._dupes = [(k, j) for k, g in enumerate(cols) for j in g[1:]]

    def _raw_score(self, text):
        """NB log-posterior from the DFA walk's sparse feature counts, None if featureless."""
        visits = visit_counts(self._dfa, text)
        if not visits:
            return None
        idx = np.fromiter(visits.keys(), dtype=np.intp, count=len(visits))
        counts = np.fromiter(visits.values(), dtype=np.float32, count=len(visits))
        return np.log1p(counts) @ self._ptc[idx] + self._model.pc

    def _word_credit(self, decoded, han):
        """Summed table credits over the distinct known tokens, per column."""
        index, indptr, cols, vals = self._words
        tokens = TOKEN_RE.findall(decoded) + (HAN_PAIR_RE.findall(decoded) if han else [])
        rows = {index[w] for w in tokens if w in index}
        if not rows:
            return None
        spans = [slice(indptr[r], indptr[r + 1]) for r in rows]
        return np.bincount(np.concatenate([cols[s] for s in spans]),
                           weights=np.concatenate([vals[s] for s in spans]),
                           minlength=len(self._model.pc))

    def _decide(self, text):
        """Scores in self.labels order, probabilities under norm_probs."""
        data, decoded = normalize(text)
        han = len(HAN_RE.findall(decoded))
        if han and 0 < len(LATIN_RE.findall(decoded)) <= LATIN_STRIP_RATIO * han:
            decoded = LATIN_RUN_RE.sub('', decoded)  # Latin in Han text pulls Mandarin to Wu
            data = decoded.encode('utf8', errors='surrogatepass')
        text = b' ' + data + b' '  # padding lets boundary n-grams fire on short input
        scores = self._raw_score(text)
        credit = self._word_credit(decoded, han)
        if scores is None:
            if credit is None:  # featureless: uniform, or RAW_FLOOR raw
                n = len(self.labels)
                return np.full(n, 1 / n if self._norm_probs else RAW_FLOOR, dtype=np.float32)
            scores = credit.astype(np.float32)
        elif credit is not None:
            scores += credit
        if self._norm_probs:
            scores *= 1.0 / math.sqrt(len(text))  # T = sqrt(bytes)
        out = scores[self._first]  # aliased label: best column, or summed as probability
        for k, j in self._dupes:
            out[k] = np.logaddexp(out[k], scores[j]) if self._norm_probs else max(out[k], scores[j])
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
