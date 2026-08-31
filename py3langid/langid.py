#!/usr/bin/env python3
"""
This file bundles language identification functions.

Modifications (fork): Copyright (c) 2021, Adrien Barbaresi.

Original code: Copyright (c) 2011 Marco Lui <saffsd@gmail.com>.
Based on research by Marco Lui and Tim Baldwin.

See LICENSE file for more info.
"""

import json
import logging
import lzma
import unicodedata
import zipfile
from collections import Counter
from http import HTTPStatus
from operator import itemgetter
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np

from .modelio import load_model as _load_model_file

LOGGER = logging.getLogger(__name__)

IDENTIFIER = None
MODEL_FILE = 'data/model.npz.xz'
MODEL_DIR = Path(__file__).parent
# raw-scale score for input with no features: finite (JSON-safe) yet far
# below any real log-probability, so it never wins an argmax or a threshold
RAW_FLOOR = float(np.finfo(np.float32).min)


def _load_identifier(model_path=None, norm_probs=False, langs=None):
    """Load an identifier: external model if given, else the bundled one."""
    identifier = None
    if model_path:
        try:
            identifier = LanguageIdentifier.from_modelpath(model_path, norm_probs=norm_probs)
            LOGGER.info("Using external model: %s", model_path)
        except (OSError, EOFError, KeyError, ValueError, lzma.LZMAError,
                zipfile.BadZipFile) as e:
            LOGGER.warning("Failed to load %s: %s", model_path, e)
    if identifier is None:
        identifier = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=norm_probs)
    if langs:
        identifier.set_languages(langs)
    return identifier


def _get_identifier():
    """Return the global identifier, loading the default model if needed."""
    global IDENTIFIER
    if IDENTIFIER is None:
        LOGGER.debug('initializing identifier')
        IDENTIFIER = _load_identifier()
    return IDENTIFIER


def set_languages(langs=None):
    """Set the language subset used by the global identifier."""
    return _get_identifier().set_languages(langs)


def classify(instance):
    """Classify a text string, returning (language, confidence)."""
    return _get_identifier().classify(instance)


def rank(instance):
    """Rank all languages by likelihood, returning [(language, confidence), ...]."""
    return _get_identifier().rank(instance)


def _init_worker(model_path, norm_probs, langs):
    # spawned Pool workers get a fresh module: rebuild the parent's identifier
    global IDENTIFIER
    IDENTIFIER = _load_identifier(model_path, norm_probs, langs)


def _process_file(path, dist=False):
    with open(path, 'rb') as f:
        text = f.read()
    return path, (rank(text) if dist else classify(text))


class LanguageIdentifier:
    __slots__ = [
        '_full_model',
        '_norm_probs',
        '_rowbase',
        'min_confidence',
        'nb_classes',
        'nb_pc',
        'nb_ptc',
        'tk_nextmove',
        'tk_output',
        'tk_row',
    ]

    @classmethod
    def from_model_file(cls, model_file, *args, **kwargs):
        "Load a model in npz+LZMA layout (relative paths resolve to the package)."
        filepath = Path(model_file)
        if not filepath.is_absolute():
            filepath = MODEL_DIR / filepath
        ptc, pc, classes, nextmove, row, output = _load_model_file(filepath)
        return cls(np.asarray(ptc), np.asarray(pc), classes, nextmove, output,
                   *args, tk_row=row, **kwargs)

    @classmethod
    def from_modelpath(cls, path, *args, **kwargs):
        "Load a model from an arbitrary path (npz+LZMA is the only layout)."
        return cls.from_model_file(Path(path).absolute(), *args, **kwargs)

    def __init__(self, nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output,
                 norm_probs=False, min_confidence=None, *, tk_row):
        # abstention: classify() returns ("und", conf) below this confidence
        if min_confidence is not None and not norm_probs:
            raise ValueError("min_confidence requires norm_probs=True")
        self.min_confidence = min_confidence
        self.nb_ptc = nb_ptc
        self.nb_pc = nb_pc
        self.nb_classes = nb_classes
        self.tk_nextmove = tk_nextmove
        # state -> row of the deduplicated transition table
        self.tk_row = tk_row
        # the walk's row offsets, pre-shifted: one list lookup per byte
        # instead of a lookup plus a shift (measured -6% per call for 3 MB)
        self._rowbase = [r << 8 for r in tk_row]
        self.tk_output = tk_output
        self._norm_probs = norm_probs
        self._full_model = nb_ptc, nb_pc, nb_classes

    @property
    def labels(self):
        "Distinct output labels; script aliases (srl->sr) share one."
        return list(dict.fromkeys(self.nb_classes))

    def set_languages(self, langs=None):
        LOGGER.debug("restricting languages to: %s", langs)
        nb_ptc, nb_pc, nb_classes = self._full_model
        if langs is None:
            self.nb_classes, self.nb_ptc, self.nb_pc = nb_classes, nb_ptc, nb_pc
        else:
            lang_set = set(langs)
            unknown = lang_set - set(nb_classes)
            if unknown:
                raise ValueError(f"Unknown language code(s): {unknown}")

            indices = [i for i, c in enumerate(nb_classes) if c in lang_set]
            self.nb_classes = [nb_classes[i] for i in indices]
            self.nb_ptc = nb_ptc[:, indices]
            self.nb_pc = nb_pc[indices]

    @staticmethod
    def _encode(text):
        if isinstance(text, bytes):
            # decode so case normalization applies uniformly to str and bytes
            try:
                text = text.decode('utf8')
            except UnicodeDecodeError:
                pass
        if isinstance(text, str):
            if text.isupper():
                text = text.lower()
            # NFC, as the training corpus (train.common.nfc_bytes)
            try:
                text = unicodedata.normalize('NFC', text)
            except ValueError:
                pass
            text = text.encode('utf8', errors='surrogatepass')
        return text

    def _sparse_score(self, visits, table):
        "NB score from a sparse {row: count} map over `table`'s rows."
        idx = np.fromiter(visits.keys(), dtype=np.intp, count=len(visits))
        counts = np.fromiter(visits.values(), dtype=np.float32, count=len(visits))
        # sublinear TF: models are trained for log1p'd counts. `table` is
        # float16 in shipped models; matmul promotes it to float32 exactly,
        # so there is nothing to gain from upcasting the whole table.
        return np.log1p(counts) @ table[idx] + self.nb_pc

    def _raw_score(self, text):
        "Raw NB scores for encoded bytes."
        # DFA walk: each state emits the one longest feature ending there
        state, indexes = 0, []
        nm, rowbase, out = self.tk_nextmove, self._rowbase, self.tk_output
        append = indexes.append
        for letter in text:
            state = nm[rowbase[state] + letter]
            f = out[state]
            if f >= 0:
                append(f)

        if indexes:
            return self._sparse_score(Counter(indexes), self.nb_ptc)

        # no features: a flat floor. Under norm_probs that is 0.0, giving a
        # uniform distribution so min_confidence abstains. On the raw scale
        # 0.0 would outrank every real (negative) score, so floor at
        # RAW_FLOOR -- finite, keeping the server's JSON valid under
        # allow_nan=False, but below anything the scorer can emit.
        fill = 0.0 if self._norm_probs else RAW_FLOOR
        return np.full(len(self.nb_classes), fill, dtype=np.float32)

    def _normalize(self, probs, nbytes):
        if self._norm_probs:
            # T = sqrt(bytes): score noise grows ~sqrt(n), keeping the
            # softmax calibrated at every length without fitted constants
            probs = probs / np.sqrt(max(nbytes, 1))
            e = np.exp(probs - probs.max())
            probs = e / e.sum()
        return probs

    def _decide(self, text):
        "Shared by classify() and rank(): one normalized score per class column."
        text = self._encode(text)
        return self._normalize(self._raw_score(text), len(text))

    def classify(self, text):
        # no per-label collapse first: a label's score is the best of its
        # columns, so the best column's label is the best label
        scores = self._decide(text)
        i = int(scores.argmax())
        conf = float(scores[i])
        if self.min_confidence is not None and conf < self.min_confidence:
            return 'und', conf
        return self.nb_classes[i], conf

    def rank(self, text):
        """Languages by likelihood, best first, one entry per label.

        Shares classify()'s decision, so rank()[0] == classify() unless
        min_confidence abstains.
        """
        ranked = sorted(zip(self.nb_classes, self._decide(text).tolist()),
                        key=itemgetter(1), reverse=True)
        best = {}
        for lang, prob in ranked:
            best.setdefault(lang, prob)  # aliased columns: keep the best one
        return list(best.items())


def _detect(data):
    lang, conf = classify(data)
    return {'language': lang, 'confidence': conf}


_ROUTES = {'detect': _detect, 'rank': rank}


def application(environ, start_response):
    """WSGI-compatible langid web service."""
    path = environ.get('PATH_INFO', '').strip('/').partition('/')[0]
    handler = _ROUTES.get(path)
    if handler is None:
        return _return_response(start_response, 404, None, 'Not found')

    method = environ['REQUEST_METHOD']
    if method not in ('GET', 'POST', 'PUT'):
        return _return_response(start_response, 405, None, f'{method} not allowed')

    data = _get_data(environ)
    if data is None:
        return _return_response(start_response, 400, None, 'No data provided')

    return _return_response(start_response, 200, handler(data), None)


def _get_data(environ):
    method = environ['REQUEST_METHOD']
    if method in ('PUT', 'POST'):
        try:
            length = int(environ.get('CONTENT_LENGTH', 0))
        except ValueError:
            return None
        if length <= 0:
            return None
        data = environ['wsgi.input'].read(length)
        if method == 'POST':
            try:
                data = parse_qs(data)[b'q'][0]
            except KeyError:
                pass
        return data
    if method == 'GET':
        try:
            return parse_qs(environ.get('QUERY_STRING', ''))['q'][0]
        except KeyError:
            return None
    return None


def _return_response(start_response, status_code, response_data, response_details):
    status = HTTPStatus(status_code)
    response = {
        'responseData': response_data,
        'responseStatus': status_code,
        'responseDetails': response_details,
    }
    headers = [('Content-type', 'application/json; charset=utf-8')]
    start_response(f"{status.value} {status.phrase}", headers)
    return [json.dumps(response).encode('utf-8')]


def main():

    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--serve', action='store_true', help='launch web service')
    parser.add_argument('--host', help='host/ip to bind to')
    parser.add_argument('--port', default=9008, type=int, help='port to listen on')
    parser.add_argument('-v', action='count', dest='verbosity', help='increase verbosity (repeat for greater effect)')
    parser.add_argument('-m', dest='model', help='load model from file')
    parser.add_argument('-l', '--langs', help='comma-separated set of target ISO639 language codes (e.g en,de)')
    parser.add_argument('-r', '--remote', action='store_true', help='auto-detect IP address for remote access')
    parser.add_argument('-b', '--batch', action='store_true', help='specify a list of files on the command line')
    parser.add_argument('-d', '--dist', action='store_true', help='show full distribution over languages')
    parser.add_argument('-u', '--url', help='langid of URL')
    parser.add_argument('--line', action='store_true', help='process pipes line-by-line rather than as a document')
    parser.add_argument('-n', '--normalize', action='store_true', help='normalize confidence scores to probability values')
    options = parser.parse_args()

    if options.verbosity:
        logging.basicConfig(level=max((5-options.verbosity)*10, 0))
    else:
        logging.basicConfig()

    if options.batch and options.serve:
        parser.error("cannot specify both batch and serve at the same time")

    global IDENTIFIER

    langs = options.langs.split(",") if options.langs else None
    IDENTIFIER = _load_identifier(options.model, options.normalize, langs)

    _process = IDENTIFIER.rank if options.dist else IDENTIFIER.classify

    if options.url:
        from urllib.request import urlopen
        with urlopen(options.url) as url:
            text = url.read()
            output = _process(text)
            print(options.url, len(text), output)

    elif options.serve:
        import socket
        from wsgiref.simple_server import make_server

        if options.remote and options.host is None:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("google.com", 80))
                hostname = s.getsockname()[0]
        elif options.host is None:
            hostname = socket.gethostbyname(socket.gethostname())
        else:
            hostname = options.host

        print(f"Listening on {hostname}:{options.port}")
        print("Press Ctrl+C to exit")
        httpd = make_server(hostname, options.port, application)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass

    elif options.batch:
        import csv
        from functools import partial
        from multiprocessing import Pool, cpu_count

        paths = [p for p in (line.strip() for line in sys.stdin)
                 if p and Path(p).is_file()]

        writer = csv.writer(sys.stdout, lineterminator='\n')
        # one model copy per worker: never spawn more workers than files
        with Pool(processes=max(1, min(cpu_count(), len(paths))),
                  initializer=_init_worker,
                  initargs=(options.model, options.normalize, langs)) as pool:
            if options.dist:
                header = IDENTIFIER.labels
                writer.writerow(['path', 'language'] + header)
                for path, ranking in pool.imap_unordered(partial(_process_file, dist=True), paths):
                    scores = dict(ranking)
                    row = [path, ranking[0][0]] + [scores[c] for c in header]
                    writer.writerow(row)
            else:
                for path, (lang, conf) in pool.imap_unordered(_process_file, paths):
                    writer.writerow((path, lang, conf))
    else:
        if sys.stdin.isatty():
            # Interactive mode
            while True:
                try:
                    print(">>>", end=' ')
                    text = input()
                except (KeyboardInterrupt, EOFError):
                    break
                print(_process(text))
        else:
            # Redirected
            if options.line:
                for line in sys.stdin:
                    print(_process(line))
            else:
                print(_process(sys.stdin.read()))


if __name__ == "__main__":
    main()
