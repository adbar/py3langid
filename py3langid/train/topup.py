"""Top-up thin classes from GlotCC / Glot500 / UDHR (see TRAINING.md)."""

import itertools
import re
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from .common import (
    ALT_CLASS,
    CLASS_SCRIPT,
    MIN_DOC,
    MIN_DOMAINS,
    SPLIT_SCRIPT,
    script_filter,
    walk_corpus,
)
from .gather_data import valid_docs, write_docs
from .sources import ISO3

GLOTCC_REPO = "cis-lmu/GlotCC-V1"
GLOT500_REPO = "cis-lmu/Glot500"
UDHR_CSV = Path("raw_downloads/udhr/udhr-lid.csv")
TOPUP_MIN_DOCS = 600
GLOT_SCRIPT = {**CLASS_SCRIPT, "crh": "Latn", "gom": "Deva"}
GLOT_ISO3 = {"uz": "uzn"}

_CONFIG_RE = re.compile(r"^[a-z]{3}([-_])[A-Z][a-z]{3}$")


def _class_iso3(cls):
    base = ALT_CLASS.get(cls, cls)
    return GLOT_ISO3.get(base) or ISO3.get(base)


def _grouped(rows, target=2000):
    """Pack small rows into ~target-byte docs; large rows pass through."""
    buf, size = [], 0
    for row in rows:
        raw = row.encode("utf-8") if isinstance(row, str) else row
        if len(raw) >= MIN_DOC:
            yield raw
        else:
            buf.append(raw)
            size += len(raw) + 1
            if size >= target:
                yield b"\n".join(buf)
                buf, size = [], 0
    if size >= MIN_DOC:
        yield b"\n".join(buf)


def _write_topup(out_dir, rows, max_docs, cls, extra_dir=None):
    keep = script_filter(cls)
    docs = (d for d in valid_docs(_grouped(rows)) if keep is None or keep(d))
    docs = itertools.islice(docs, max_docs * 5)
    if out_dir.is_dir() and any(out_dir.glob("*.txt")):
        n = sum(1 for _ in itertools.islice(docs, max_docs))  # skip past primary
    else:
        n = write_docs(out_dir, docs, max_docs)
    if extra_dir is not None:
        # extra_dir is the resume marker: write to a tmp dir and rename on
        # success so a mid-stream failure never marks the class as done
        tmp = extra_dir.with_name(extra_dir.name + ".tmp")
        if tmp.is_dir():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        for i, d in enumerate(docs):
            (tmp / f"doc{i:04d}.txt").write_bytes(d)
        tmp.rename(extra_dir)
    return n


def _repo_configs(repo):
    from huggingface_hub import list_repo_files
    configs = defaultdict(set)
    for f in list_repo_files(repo, repo_type="dataset"):
        for comp in f.split("/"):
            if _CONFIG_RE.match(comp):
                configs[comp[:3]].add(comp)
                break
    return configs


def _glot_config(configs, cls):
    cands = sorted(configs.get(_class_iso3(cls) or "", ()))
    if len(cands) > 1:
        want = GLOT_SCRIPT.get(cls)
        cands = [c for c in cands if want and c.endswith(want)]
    return cands[0] if len(cands) == 1 else None


def _stream_hf(repo, config, field):
    from datasets import load_dataset
    ds = load_dataset(repo, config, split="train", streaming=True)
    return (row[field] for row in ds)


def _topup_class(out_root, extra_root, repo, source, field, configs, max_docs, cls):
    out_dir = out_root / source / cls
    extra_dir = extra_root / source / cls
    if extra_dir.is_dir():
        return  # resume: fetched with extras kept
    config = _glot_config(configs, cls)
    if not config:
        print(f"{source}/{cls}: no config")
        return
    try:
        n = _write_topup(out_dir, _stream_hf(repo, config, field),
                         max_docs, cls, extra_dir)
        extra = sum(1 for _ in extra_dir.glob("*.txt"))
        print(f"{source}/{cls}: {n} docs (+{extra} extra) ({config})")
    except Exception as e:
        print(f"{source}/{cls}: SKIP ({e})")


def class_counts(out_root):
    counts = defaultdict(lambda: defaultdict(int))
    for domain, cls, _path in walk_corpus(out_root):
        counts[cls][domain] += 1
    return counts


def gather_udhr(out_root, classes, max_docs):
    """UDHR sentences (legal register) for classes the CSV covers."""
    if not UDHR_CSV.exists():
        print(f"udhr: {UDHR_CSV} missing, skipped")
        return
    import csv
    rows = defaultdict(list)
    with open(UDHR_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["iso639-3"]].append(row["sentence"])
    for cls in classes:
        sents = rows.get(_class_iso3(cls) or "", ())
        if sents:
            n = _write_topup(out_root / "udhr" / cls, sents, max_docs, cls)
            if n:
                print(f"udhr/{cls}: {n} docs")


def gather_topup(out_root, langs, max_docs, jobs=4):
    classes = []
    for lang in langs:
        classes.append(lang)
        if lang in SPLIT_SCRIPT:
            classes.append(SPLIT_SCRIPT[lang].alt)

    def needy():
        """Classes below TOPUP_MIN_DOCS or MIN_DOMAINS."""
        counts = class_counts(out_root)
        return [c for c in classes
                if sum(counts[c].values()) < TOPUP_MIN_DOCS
                or len(counts[c]) < MIN_DOMAINS]

    extra_root = out_root.with_name(out_root.name + "_extra")
    thin = needy()
    print(f"topup: {len(thin)} thin classes: {thin}")
    for repo, source, field in ((GLOTCC_REPO, "glotcc", "content"),
                                (GLOT500_REPO, "glot500", "text")):
        configs = _repo_configs(repo)
        one = partial(_topup_class, out_root, extra_root, repo, source, field,
                      configs, max_docs)
        gathered = ([p.name for p in (out_root / source).iterdir() if p.is_dir()]
                    if (out_root / source).is_dir() else [])
        with ThreadPoolExecutor(jobs) as pool:
            list(pool.map(one, sorted(set(thin) | set(gathered))))
        thin = needy()  # this repo just wrote docs
    if thin:
        gather_udhr(out_root, thin, max_docs)
    print(f"topup done; still thin: {thin}")
