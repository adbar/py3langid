"""Gather a multi-domain training corpus (see TRAINING.md).

Output layout: OUTPUT/{tatoeba,cc100,wiki,leipzig}/{lang}/docNNNN.txt (UTF-8 bytes).
"""

import argparse
import bz2
import io
import json
import lzma
import re
import shutil
import tarfile
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..modelio import load_model
from .common import SPLIT_SCRIPT, route_script
from .sources import CC100_CODE, ISO3, LEIPZIG_NAME, WIKI_CODE

TATOEBA_URL = "https://downloads.tatoeba.org/exports/sentences.tar.bz2"
CC100_URL = "https://data.statmt.org/cc-100/{code}.txt.xz"
CIRRUS_INDEX = "https://dumps.wikimedia.org/other/cirrus_search_index/"
CIRRUS_URL = CIRRUS_INDEX + "{date}/index_name%3D{code}wiki_content/{code}wiki_content-{date}-00000.json.bz2"
LEIPZIG_URL = "https://downloads.wortschatz-leipzig.de/corpora/{name}.tar.gz"

MIN_DOC = 500       # skip stubs
MAX_DOC = 3_000     # truncate long docs; equalizes byte weight across domains
CC100_RANGE = 2 * 1024 * 1024
RAW_CACHE = Path("raw_downloads")  # downloads kept on disk, reused on re-gather

USER_AGENT = "py3langid-gather/0.1 (https://github.com/adbar/py3langid)"


def fetch(url, headers=None, retries=3):
    # wikimedia rejects the default urllib User-Agent
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
    """Download url fully to cache_path once; return an open binary handle."""
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_name(cache_path.name + ".tmp")
        with fetch(url, headers) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.replace(cache_path)
    return open(cache_path, "rb")


class _TeeReader:
    """Mirrors consumed response bytes to a cache file; kept only on finalize()."""

    def __init__(self, resp, cache_path):
        self.resp = resp
        self.path = cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = cache_path.with_name(cache_path.name + ".tmp")
        self.f = open(self.tmp, "wb")  # noqa: SIM115

    def read(self, n=-1):
        chunk = self.resp.read(n)
        self.f.write(chunk)
        return chunk

    def finalize(self):
        self.f.close()
        self.resp.close()
        self.tmp.replace(self.path)

    def discard(self):
        self.f.close()
        self.resp.close()
        self.tmp.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finalize() if exc_type is None else self.discard()


def model_langs():
    path = Path(__file__).parent.parent / "data" / "model.npz.xz"
    return list(load_model(path)[2])


class DocWriter:
    """Routes docs by script, numbers files, enforces per-dir caps."""

    def __init__(self, out_dir, max_docs):
        self.out_dir = out_dir
        self.lang = out_dir.name
        self.max_docs = max_docs
        self.counts = defaultdict(int)

    def write(self, doc):
        d = route_script(self.out_dir, doc)
        if self.counts[d.name] < self.max_docs:
            d.mkdir(parents=True, exist_ok=True)
            (d / f"doc{self.counts[d.name]:04d}.txt").write_bytes(doc)
            self.counts[d.name] += 1

    @property
    def done(self):
        """For split-script langs, done when both halves are full."""
        spec = SPLIT_SCRIPT.get(self.lang)
        if spec:
            return min(self.counts[self.lang], self.counts[spec.alt]) >= self.max_docs
        return self.counts[self.lang] >= self.max_docs

    @property
    def total(self):
        return sum(self.counts.values())


def valid_docs(docs):
    """The one doc-validity gate: strip, truncate, drop stubs."""
    for doc in docs:
        doc = doc.strip()[:MAX_DOC]
        if len(doc) >= MIN_DOC:
            yield doc


def write_docs(out_dir, docs, max_docs):
    w = DocWriter(out_dir, max_docs)
    for doc in valid_docs(docs):
        w.write(doc)
        if w.done:
            break
    return w.total


def gather_cc100(out_root, lang, max_docs):
    code = CC100_CODE.get(lang, lang)
    with fetch_cached(CC100_URL.format(code=code),
                      RAW_CACHE / "cc100" / f"{code}.txt.xz.head{CC100_RANGE}",
                      {"Range": f"bytes=0-{CC100_RANGE - 1}"}) as resp:
        data = resp.read()
    # tolerant decompression of the truncated .xz prefix: keep partial output
    out = []
    dec = lzma.LZMADecompressor()
    try:
        for i in range(0, len(data), 1 << 16):
            out.append(dec.decompress(data[i:i + (1 << 16)]))
    except lzma.LZMAError:
        pass
    docs = b"".join(out).split(b"\n\n")[:-1]  # last doc may be cut
    return write_docs(out_root / "cc100" / lang, docs, max_docs)


def gather_wiki(out_root, lang, max_docs, date):
    code = WIKI_CODE.get(lang, lang)
    # cirrus dumps vanish upstream: keep the consumed .bz2 prefix on disk.
    # The prefix is only as long as the run that wrote it needed, so the doc
    # target is part of the name and only a >= prefix may be reused -- else a
    # later, larger run would hit EOF early and silently gather too little.
    stem = f"{code}wiki-{date}.json.bz2.head"
    cache = RAW_CACHE / "wiki" / f"{stem}{max_docs}"
    usable = [p for p in sorted(cache.parent.glob(f"{stem}*"))
              if p.name[len(stem):].isdigit()
              and int(p.name[len(stem):]) >= max_docs]
    if usable:
        resp = open(usable[0], "rb")  # noqa: SIM115
    else:
        resp = _TeeReader(fetch(CIRRUS_URL.format(code=code, date=date)), cache)
    docs = []
    dec = bz2.BZ2Decompressor()
    buf = b""
    read = 0
    with resp:
        while len(docs) < max_docs and read < (1 << 28):  # 256 MiB safety cap
            chunk = resp.read(1 << 18)
            if not chunk:
                break
            read += len(chunk)
            buf += dec.decompress(chunk)
            *lines, buf = buf.split(b"\n")
            for line in lines:
                if b'"text"' not in line:
                    continue
                text = json.loads(line).get("text")
                if text:
                    doc = text.encode("utf-8").strip()
                    if len(doc) >= MIN_DOC:  # filter stubs while reading
                        docs.append(doc)
    return write_docs(out_root / "wiki" / lang, docs, max_docs)


def _writer_counts(writers):
    return {name: n for w in writers.values() for name, n in w.counts.items() if n}


def gather_tatoeba(out_root, langs, max_docs, per_doc):
    by_iso3 = {ISO3[lang]: lang for lang in langs if lang in ISO3}
    remaining = set(by_iso3.values())
    buf = defaultdict(list)
    writers = {}
    resp = fetch_cached(TATOEBA_URL, RAW_CACHE / "tatoeba" / "sentences.tar.bz2")
    with tarfile.open(fileobj=resp, mode="r|bz2") as tar:
        for member in tar:
            if not member.name.endswith("sentences.csv"):
                continue
            # no TextIOWrapper: the streamed tar member is not seekable
            for raw in tar.extractfile(member):
                parts = raw.decode("utf-8").rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                lang = by_iso3.get(parts[1])
                if lang is None or lang not in remaining:
                    continue
                buf[lang].append(parts[2])
                if len(buf[lang]) == per_doc:
                    doc = "\n".join(buf[lang]).encode("utf-8")
                    buf[lang] = []
                    if lang not in writers:
                        writers[lang] = DocWriter(out_root / "tatoeba" / lang, max_docs)
                    writers[lang].write(doc)
                    if writers[lang].done:
                        remaining.discard(lang)
                        if not remaining:
                            return _writer_counts(writers)
    return _writer_counts(writers)



def gather_leipzig(out_root, lang, max_docs, per_doc):
    name = LEIPZIG_NAME.get(lang)
    if not name:
        return 0
    with fetch_cached(LEIPZIG_URL.format(name=name),
                      RAW_CACHE / "leipzig" / f"{name}.tar.gz") as resp:
        data = resp.read()
    sentences = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar:
            if not member.name.endswith("-sentences.txt"):
                continue
            for raw in tar.extractfile(member):
                parts = raw.decode("utf-8", errors="replace").rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    sentences.append(parts[1])
    docs = ("\n".join(sentences[i:i + per_doc]).encode("utf-8")
            for i in range(0, len(sentences), per_doc))
    return write_docs(out_root / "leipzig" / lang, docs, max_docs)


def latest_cirrus_date():
    html = fetch(CIRRUS_INDEX).read().decode()
    dates = sorted(set(re.findall(r'href="(\d{8})/"', html)))
    # newest dir may be an in-progress dump
    return dates[-2] if len(dates) > 1 else dates[-1]


def per_lang_domain(name, func, langs, jobs, out_root, max_docs):
    # resume: skip languages already gathered (glob on a missing dir is empty)
    todo = [lang for lang in langs
            if sum(1 for _ in (out_root / name / lang).glob("*.txt")) < max_docs]
    if len(todo) < len(langs):
        print(f"{name}: {len(langs) - len(todo)} langs already complete")
    counts = {}

    def one(lang):
        try:
            counts[lang] = func(lang)
        except Exception as e:
            print(f"{name}/{lang}: SKIP ({e})")

    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(one, todo))
    for lang in sorted(counts):
        print(f"{name}/{lang}: {counts[lang]} docs")
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="corpus output directory")
    parser.add_argument("--langs", help="comma-separated ISO 639-1 codes (default: the 97 model languages)")
    parser.add_argument("--domains", default="tatoeba,cc100,wiki,leipzig", help="comma-separated subset of domains")
    parser.add_argument("--max-docs-per-lang", type=int, default=300)
    parser.add_argument("--sentences-per-doc", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=4, help="parallel downloads (cc100/wiki)")
    parser.add_argument("--wiki-date", help="cirrus dump date YYYYMMDD (default: latest complete)")
    args = parser.parse_args(argv)

    langs = args.langs.split(",") if args.langs else model_langs()
    domains = args.domains.split(",")
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    max_docs = args.max_docs_per_lang

    if "cc100" in domains:
        per_lang_domain("cc100", lambda lang: gather_cc100(out_root, lang, max_docs), langs, args.jobs, out_root, max_docs)
    if "wiki" in domains:
        date = args.wiki_date or latest_cirrus_date()
        print(f"wiki: cirrus dump {date}")
        per_lang_domain("wiki", lambda lang: gather_wiki(out_root, lang, max_docs, date), langs, args.jobs, out_root, max_docs)
    if "tatoeba" in domains:
        counts = gather_tatoeba(out_root, langs, max_docs, args.sentences_per_doc)
        print(f"tatoeba: {len(counts)} langs")
        for lang in sorted(set(langs) - set(counts)):
            print(f"tatoeba/{lang}: 0 docs")
    if "leipzig" in domains:
        per_lang_domain("leipzig",
                        lambda lang: gather_leipzig(out_root, lang, max_docs, args.sentences_per_doc),
                        langs, args.jobs, out_root, max_docs)
    if "topup" in domains:
        from .topup import gather_topup
        gather_topup(out_root, langs, max_docs, args.jobs)


if __name__ == "__main__":
    main()
