"""Gather a multi-domain training corpus (see TRAINING.md)."""

import argparse
import bz2
import hashlib
import json
import lzma
import re
import shutil
import tarfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from .. import __version__
from . import topup
from .clean import clean as clean_corpus
from .common import ALT_CLASS, SENT_SPLIT, SPLIT_SCRIPT
from .sources import CC100_CODE, ISO3, LEIPZIG_NAME, WIKI_CODE
from .writer import PACK_TARGET, gather_domain, pack_docs

TATOEBA_URL = "https://downloads.tatoeba.org/exports/sentences.tar.bz2"
CC100_URL = "https://data.statmt.org/cc-100/{code}.txt.xz"
CIRRUS_INDEX = "https://dumps.wikimedia.org/other/cirrus_search_index/"
CIRRUS_URL = CIRRUS_INDEX + "{date}/index_name%3D{code}wiki_content/{code}wiki_content-{date}-00000.json.bz2"
LEIPZIG_URL = "https://downloads.wortschatz-leipzig.de/corpora/{name}.tar.gz"

CC100_RANGE = 2 * 1024 * 1024
LEIPZIG_SENTS = 50  # sentences per Leipzig doc, capped at DOC_CAP
WIKI_DOCS_FACTOR = 3  # wiki leads are short
WIKI_RANGE = 64 * 1024 * 1024  # ~20 KB of dump per lead
RAW_CACHE = Path("raw_downloads")  # downloads kept on disk, reused on re-gather
USED_RAW = set()  # cache files read by this gather, for the manifest

USER_AGENT = "py3langid-gather/0.1 (https://github.com/adbar/py3langid)"

URL = re.compile(r"https?://\S+")


def fetch(url, headers=None, retries=3):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as e:
            if e.code == 404 or attempt == retries - 1:
                raise
            time.sleep(60 if e.code == 429 else 5)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5)


def fetch_cached(url, cache_path, headers=None):
    """Download to cache_path once; return open binary handle."""
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_name(cache_path.name + ".tmp")
        with fetch(url, headers) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.replace(cache_path)
    USED_RAW.add(cache_path)
    return open(cache_path, "rb")


def fetch_head(url, cache_path, size):
    """The first *size* bytes of url, cached."""
    return fetch_cached(url, cache_path.with_name(f"{cache_path.name}.head{size}"),
                        {"Range": f"bytes=0-{size - 1}"})


def cc100_docs(lang):
    code = CC100_CODE.get(lang, lang)
    with fetch_head(CC100_URL.format(code=code), RAW_CACHE / "cc100" / f"{code}.txt.xz",
                    CC100_RANGE) as resp:
        data = lzma.LZMADecompressor().decompress(resp.read())  # truncated input decodes without error
    return data.split(b"\n\n")[:-1]


def novel_sentences(text, seen):
    """Unseen sentences, one per line (bot stubs repeat template sentences)."""
    out = []
    for sent in SENT_SPLIT.split(URL.sub("", text)):
        sent = " ".join(sent.split())
        if sent and sent not in seen:
            seen.add(sent)
            out.append(sent)
    return "\n".join(out).encode("utf-8")


def _wiki_leads(resp):
    """Yield opening_text per article (full "text" carries reference sections in other languages)."""
    dec = bz2.BZ2Decompressor()
    buf = b""
    while chunk := resp.read(1 << 18):
        buf += dec.decompress(chunk)
        *lines, buf = buf.split(b"\n")
        for line in lines:
            if b'"opening_text"' in line:
                text = json.loads(line).get("opening_text")
                if text:
                    yield text


def wiki_docs(lang, date):
    code = WIKI_CODE.get(lang, lang)
    seen = set()
    with fetch_head(CIRRUS_URL.format(code=code, date=date),
                    RAW_CACHE / "wiki" / f"{code}wiki-{date}.json.bz2", WIKI_RANGE) as resp:
        for text in _wiki_leads(resp):
            yield novel_sentences(text, seen)


def tatoeba_docs(langs, max_docs):
    """{lang: packed docs}, up to max_docs per lang."""
    by_iso3 = {ISO3[lang].encode(): lang for lang in langs if lang in ISO3}
    budget = max_docs * PACK_TARGET  # raw bytes kept per lang before packing
    rows, size = defaultdict(list), defaultdict(int)
    with fetch_cached(TATOEBA_URL, RAW_CACHE / "tatoeba" / "sentences.tar.bz2") as resp, \
         tarfile.open(fileobj=resp, mode="r|bz2") as tar:
        for member in tar:
            if not member.name.endswith("sentences.csv"):
                continue
            for raw in tar.extractfile(member):
                parts = raw.rstrip(b"\n").split(b"\t")
                if len(parts) != 3:
                    continue
                lang = by_iso3.get(parts[1])
                if lang is not None and size[lang] < budget:
                    rows[lang].append(parts[2])
                    size[lang] += len(parts[2]) + 1
    return {lang: list(pack_docs(sents))[:max_docs] for lang, sents in rows.items()}


def leipzig_docs(lang):
    name = LEIPZIG_NAME.get(lang)
    if not name:
        return
    sentences = []
    with fetch_cached(LEIPZIG_URL.format(name=name),
                      RAW_CACHE / "leipzig" / f"{name}.tar.gz") as resp, \
         tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            if not member.name.endswith("-sentences.txt"):
                continue
            for raw in tar.extractfile(member):
                parts = raw.decode("utf-8", errors="replace").rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    sentences.append(parts[1])
    # file order (alphabetical) and fixed-size chunks reproduce the release corpus,
    # a uniform sample measured worse on CommonLID (-0.07)
    for i in range(0, len(sentences), LEIPZIG_SENTS):
        yield "\n".join(sentences[i:i + LEIPZIG_SENTS]).encode("utf-8")


def latest_cirrus_date():
    html = fetch(CIRRUS_INDEX).read().decode()
    dates = sorted(set(re.findall(r'href="(\d{8})/"', html)))
    if not dates:
        raise RuntimeError(f"no dumps listed at {CIRRUS_INDEX}; pass --wiki-date")
    return dates[-2] if len(dates) > 1 else dates[-1]



def write_manifest(out_root, info):
    """MANIFEST.json: settings of every gather run and the raw files read."""
    path = out_root / "MANIFEST.json"
    old = json.loads(path.read_text("utf-8")) if path.exists() else {}
    raw = old.get("raw", {})
    for p in sorted(USED_RAW):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        raw[str(p.relative_to(RAW_CACHE))] = {"size": p.stat().st_size, "sha256": h.hexdigest()}
    runs = old.get("runs", []) + [{**info, "version": __version__}]
    path.write_text(json.dumps({"runs": runs, "raw": raw}, indent=1) + "\n", "utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="corpus output directory")
    parser.add_argument("--langs", help="comma-separated language codes (default: every ISO3 entry)")
    parser.add_argument("--domains", default="tatoeba,cc100,wiki,leipzig", help="comma-separated subset of domains")
    parser.add_argument("--max-docs-per-lang", type=int, default=300, help="cap per language per domain, tripled for wiki (default: 300)")
    parser.add_argument("--jobs", type=int, default=4, help="parallel downloads (cc100/wiki)")
    parser.add_argument("--wiki-date", help="cirrus dump date YYYYMMDD (default: latest complete)")
    args = parser.parse_args(argv)

    langs = args.langs.split(",") if args.langs else list(ISO3)
    domains = args.domains.split(",")
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    max_docs = args.max_docs_per_lang
    USED_RAW.clear()

    if "cc100" in domains:
        # zh-Hans and zh-Hant are separate cc100 files: no routing, or the two writers collide
        own = {c for c in CC100_CODE if ALT_CLASS.get(c, c) in langs}
        gather_domain("cc100", cc100_docs, sorted(set(langs) | own), args.jobs, out_root,
                      max_docs, no_split=own)
    info = {"langs": langs, "max_docs_per_lang": max_docs}
    if "wiki" in domains:
        date = info["wiki_date"] = args.wiki_date or latest_cirrus_date()
        print(f"wiki: cirrus dump {date}")
        gather_domain("wiki", lambda lang: wiki_docs(lang, date), langs, args.jobs, out_root,
                      WIKI_DOCS_FACTOR * max_docs)
    if "tatoeba" in domains:
        docs = tatoeba_docs(langs, 3 * max_docs)  # spares for doc-rule skips and the wuu quota
        for lang in SPLIT_SCRIPT.keys() & docs.keys():  # both script classes share one budget
            docs[lang] = docs[lang][:max_docs]
        gather_domain("tatoeba", lambda lang: docs.get(lang, ()), langs, 1, out_root, max_docs)
    if "leipzig" in domains:
        gather_domain("leipzig", leipzig_docs, langs, args.jobs, out_root, max_docs)
    if "topup" in domains:
        clean_corpus(out_root)  # the gate counts cleaned docs
        topup.gather_topup(out_root, langs, max_docs, args.jobs)
        info["hf_revisions"] = topup.REVISION
    clean_corpus(out_root)  # a resumed gather rewrites cells under their cap
    write_manifest(out_root, info)


if __name__ == "__main__":
    main()
