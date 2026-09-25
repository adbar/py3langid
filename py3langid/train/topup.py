"""Top-up thin classes from GlotCC, and a few classes from Glot500 (see TRAINING.md)."""

import re
from collections import defaultdict
from functools import partial

from .common import ALT_CLASS, walk_corpus
from .sources import ISO3
from .writer import gather_domain, pack_docs

GLOTCC_REPO = "cis-lmu/GlotCC-V1"
GLOT500_REPO = "cis-lmu/Glot500"
REVISION = {GLOTCC_REPO: "9ad140b6be3ac7b539606a2b4809b49d122823de",
            GLOT500_REPO: "f7b3e9d11a7e974ec0d021342cd8b41fd67510c1"}
EVAL_SOURCES = frozenset({"Flores200"})  # Glot500 sub-corpora that are eval sets
# elsewhere Glot500 is noise (sn Kinyarwanda, pcm date lists)
GLOT500_CLASSES = ("ace", "kik", "nso", "rw", "st")
TOPUP_MIN_DOMAINS = 3  # domains with at least TOPUP_DOMAIN_DOCS docs
TOPUP_DOMAIN_DOCS = 50
TOPUP_SKIP = frozenset({"pcm"})  # GlotCC pcm draws English to pcm
GLOT_SCRIPT = {"crh": "Latn", "gom": "Deva", "sr": "Cyrl", "srl": "Latn",
               "uz": "Latn", "uzc": "Cyrl", "zh": "Hans", "zht": "Hant"}
GLOT_ISO3 = {"uz": "uzn"}

_CONFIG_RE = re.compile(r"^[a-z]{3}([-_])[A-Z][a-z]{3}$")


def _class_iso3(cls):
    base = ALT_CLASS.get(cls, cls)
    return GLOT_ISO3.get(base) or ISO3.get(base)


def _repo_configs(repo):
    from huggingface_hub import list_repo_files
    configs = defaultdict(set)
    for f in list_repo_files(repo, repo_type="dataset", revision=REVISION[repo]):
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


def _glot_docs(repo, field, configs, cls):
    config = _glot_config(configs, cls)
    if not config:
        raise LookupError("no config")
    from datasets import load_dataset
    ds = load_dataset(repo, config, split="train", streaming=True, revision=REVISION[repo])
    return pack_docs(row[field] for row in ds if row.get("dataset") not in EVAL_SOURCES)


def class_counts(out_root):
    counts = defaultdict(lambda: defaultdict(int))
    for domain, cls, _path in walk_corpus(out_root):
        counts[cls][domain] += 1
    return counts


def needy(out_root, classes):
    """Classes with fewer than TOPUP_MIN_DOMAINS domains of TOPUP_DOMAIN_DOCS docs."""
    counts = class_counts(out_root)
    return [c for c in classes if c not in TOPUP_SKIP
            and sum(n >= TOPUP_DOMAIN_DOCS for n in counts[c].values()) < TOPUP_MIN_DOMAINS]


def gather_topup(out_root, langs, max_docs, jobs=4):
    classes = langs + [alt for alt, lang in ALT_CLASS.items() if lang in langs]
    thin = needy(out_root, classes)
    print(f"topup: {len(thin)} thin classes: {thin}")
    for repo, source, field, targets in (
            (GLOTCC_REPO, "glotcc", "content", thin),
            (GLOT500_REPO, "glot500", "text", [c for c in GLOT500_CLASSES if c in classes])):
        configs = _repo_configs(repo)
        # never re-stream a class this source already touched
        todo = [c for c in targets if not any((out_root / source / c).glob("*.txt"))]
        gather_domain(source, partial(_glot_docs, repo, field, configs), todo,
                      jobs, out_root, max_docs, no_split=todo)
    print(f"topup done; still thin: {needy(out_root, classes)}")
