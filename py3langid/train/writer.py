"""Write gathered docs into corpus cells."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from .common import ALT_CLASS, MIN_DOC, SPLIT_SCRIPT, cap_bytes, class_of, hant_majority
from .rules import DOC_RULES

PACK_TARGET = 2000  # bytes per doc when packing sentences

# minority-script docs admitted past the gather cap, up to max_docs/2 (Wu wiki is 15% Traditional)
SCRIPT_QUOTA = {"wuu": hant_majority}


class DocWriter:
    """Number files into out_dir under per-dir caps and script quotas; split-script
    langs route to the alt dir, split=False skips the other script instead.
    Docs failing their class doc rule are skipped, so cells fill with passing docs."""

    def __init__(self, out_dir, max_docs, split=True):
        self.out_dir = out_dir
        self.lang = out_dir.name
        self.max_docs = max_docs
        self.alt = SPLIT_SCRIPT.get(self.lang, (None,))[0] if split else None
        self.quota = SCRIPT_QUOTA.get(self.lang)
        self.quota_left = max_docs // 2 if self.quota else 0
        self.counts = defaultdict(int)

    def write(self, doc):
        cls = class_of(ALT_CLASS.get(self.lang, self.lang), doc)
        if self.alt is None and cls != self.lang:
            return
        rule = DOC_RULES.get(cls)
        if rule and rule(doc.decode("utf-8", "ignore")):
            return
        d = self.out_dir.with_name(cls)
        if self.counts[cls] >= self.max_docs:
            if not (self.quota_left and self.quota(doc)):
                return
            self.quota_left -= 1
        d.mkdir(parents=True, exist_ok=True)
        (d / f"doc{self.counts[cls]:04d}.txt").write_bytes(doc)
        self.counts[cls] += 1

    def fill(self, docs):
        """Write valid docs until done."""
        for doc in map(valid_doc, docs):
            if doc is None:
                continue
            self.write(doc)
            if self.done:
                break
        return self.total

    @property
    def done(self):
        if self.quota_left:  # reads on for minority docs until the quota fills or the source ends
            return False
        if self.alt:
            return min(self.counts[self.lang], self.counts[self.alt]) >= self.max_docs
        return self.counts[self.lang] >= self.max_docs

    @property
    def total(self):
        return sum(self.counts.values())


def valid_doc(doc):
    """Strip, cap, drop stubs. Returns bytes or None."""
    doc = cap_bytes(doc.strip())
    return doc if len(doc) >= MIN_DOC else None


def pack_docs(rows, target=PACK_TARGET):
    """Pack small rows (str or bytes) into ~target-byte docs; large rows pass through.
    A row of MIN_DOC bytes passes alone, so non-Latin scripts get short docs.
    Packing everything to DOC_CAP was measured worse on CommonLID (-0.13)."""
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


def write_docs(out_dir, docs, max_docs, split=True):
    return DocWriter(out_dir, max_docs, split).fill(docs)


def _doc_count(domain_dir, lang):
    # only the primary dir gates completion: minority-script dirs may never
    # fill from mono-script sources (topup covers them)
    return sum(1 for _ in (domain_dir / lang).glob("*.txt"))


def gather_domain(name, docs_of, langs, jobs, out_root, max_docs, no_split=()):
    """Write docs_of(lang) for every lang whose dir is not full yet.
    no_split: langs whose source is already script-pure (no routing to the alt dir)."""
    todo = [lang for lang in langs
            if _doc_count(out_root / name, lang) < max_docs]
    if len(todo) < len(langs):
        print(f"{name}: {len(langs) - len(todo)} langs already complete")
    counts = {}

    def one(lang):
        try:
            counts[lang] = write_docs(out_root / name / lang, docs_of(lang), max_docs,
                                      split=lang not in no_split)
        except Exception as e:  # noqa: BLE001
            print(f"{name}/{lang}: SKIP ({e})")

    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(one, todo))
    for lang in sorted(counts):
        print(f"{name}/{lang}: {counts[lang]} docs")
    return counts
