"""Class rules for sources that carry another language or variety (see TRAINING.md).

Doc rules apply at gather time. Line rules blank lines of
MIN_PARA bytes or more and drop a doc left under MIN_DOC.
"""
import math
import re
from collections import Counter
from pathlib import Path

from ..langid import TOKEN_RE, normalize
from .common import MIN_DOC, drop, walk_corpus

HEBREW_LETTER = re.compile("[א-ת]")
HEBREW_POINT = re.compile("[֑-ׇ]")  # cantillation and vowel points
MIN_POINT_RATIO = 0.1
EGYPTIAN = frozenset((
    "مش", "عايز", "عاوز", "عايزة", "ازاي", "إزاي", "دلوقتي", "دلوقت", "كده", "كدا", "علشان", "عشان",
    "بتاع", "بتاعة", "بتوع", "امبارح", "النهارده", "مفيش", "خالص", "اوي", "أوي", "ايه", "إيه",
    "بقى", "ليه", "احنا", "إحنا", "انتو", "دول", "برده",
))
CANTONESE = frozenset("嘅係咗冇唔喺啲佢哋嚟")
DEVANAGARI = re.compile("[ऀ-ॿ]")
MIN_PARA = 150


def unpointed(text):
    """Hebrew-script text with fewer points than MIN_POINT_RATIO per letter."""
    letters = len(HEBREW_LETTER.findall(text))
    return not letters or len(HEBREW_POINT.findall(text)) / letters < MIN_POINT_RATIO


def not_egyptian(text):
    return EGYPTIAN.isdisjoint(TOKEN_RE.findall(text))


def not_cantonese(text):
    return CANTONESE.isdisjoint(text)


def not_devanagari(line):
    """Devanagari is less than half of the letters."""
    text = line.decode("utf-8", "ignore")
    letters = sum(ch.isalpha() for ch in text)
    return letters > 0 and 2 * len(DEVANAGARI.findall(text)) < letters


def _words(data):
    return TOKEN_RE.findall(normalize(data)[1])


def zulu_test(corpus):
    """cc100 xh line test: Zulu-vs-Xhosa word log-ratio > 0, Xhosa counted outside cc100."""
    xh, zu = Counter(), Counter()
    for domain, lang, p in walk_corpus(corpus):
        if lang == "zu" or (lang == "xh" and domain != "cc100"):
            (zu if lang == "zu" else xh).update(_words(Path(p).read_bytes()))
    if not xh or not zu:  # nothing to compare against: keep every line
        return lambda line: False
    vocab = len(set(xh) | set(zu))
    nx, nz = sum(xh.values()) + 0.5 * vocab, sum(zu.values()) + 0.5 * vocab
    return lambda line: sum(math.log((zu[w] + 0.5) / nz) - math.log((xh[w] + 0.5) / nx)
                            for w in _words(line)) > 0


DOC_RULES = {"hbo": unpointed, "arz": not_egyptian, "yue": not_cantonese}


def apply_rules(corpus):
    """Line rules, in place. Returns ({lang: docs dropped}, {lang: bytes stripped})."""
    zulu = zulu_test(corpus)
    dropped, stripped, dead = Counter(), Counter(), []
    for domain, lang, path in walk_corpus(corpus):
        if lang == "gom":
            test = not_devanagari
        elif (lang, domain) == ("xh", "cc100"):
            test = zulu
        else:
            continue
        lines, n = [], 0
        for line in Path(path).read_bytes().split(b"\n"):
            if len(line) >= MIN_PARA and test(line):
                n += len(line)
                line = b""
            lines.append(line)
        if not n:
            continue
        stripped[lang] += n
        out = b"\n".join(lines)
        if len(out) < MIN_DOC:
            dead.append(path)
            dropped[lang] += 1
        else:
            Path(path).write_bytes(out)
    drop(corpus, dead)
    return dict(dropped), dict(stripped)
