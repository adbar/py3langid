"""Training pipeline constants and helpers."""

import multiprocessing as mp
import re
import sys
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

from ..langid import normalize

MAX_NGRAM_ORDER = 5
MIN_NGRAM_ORDER = 2
DF_TOKENS = 60000  # candidate pool per order
FEATURES_PER_LANG = 1600  # per-language, not global (keeps script-novel langs viable)
COUNT_FLOOR = 3  # NB counts at or below this are zeroed before smoothing
CELL_COST = 3  # LD discount per share of classes above COUNT_FLOOR (model cells)
DOC_CAP = 3000  # byte budget: gathering, tokenization, zxx
MIN_DOC = 500
NORMALIZE_VERSION = 8  # bump when normalize or the shard payload changes

Item = tuple[str, str, str]  # (domain, lang, path)
R = TypeVar("R")

SENT_SPLIT = re.compile(r"(?<=[.!?])(?:\s+|(?=[　-鿿＀-￯]))|(?<=[。！？।])")


# Traditional vs Simplified Chinese: 150 frequent, script-pure character pairs
_HANT = frozenset(
    "為時來會個這過對發現開經還們當與說動間進實國關體沒點將內讓從樣機無長麼應業場種學兩問別給結題產網幾設帶數務變電認該計總覺話區資選強處記頭東達員報爲見箇氣單傳旹門據統專確髮論導滿風許備質觀萬視難標準類則調連約續較決運請夠費參規際卻愛邊辦線級驗歡轉創議領隨雖價術離顯師組裝書項識車樂買況聯團張獲優態熱節圖"
)
_HANS = frozenset(
    "为时来会个这过对发现开经还们当与说动间进实国关体没点将内让从样机无长么应业场种学两问别给结题产网几设带数务变电认该计总觉话区资选强处记头东达员报为见个气单传时门据统专确发论导满风许备质观万视难标准类则调连约续较决运请够费参规际却爱边办线级验欢转创议领随虽价术离显师组装书项识车乐买况联团张获优态热节图"
)


def hant_majority(doc: bytes) -> bool:
    """True if doc has more Traditional than Simplified marker characters."""
    text = doc.decode("utf-8", errors="surrogateescape")
    return sum(ch in _HANT for ch in text) > sum(ch in _HANS for ch in text)


def latin_majority(doc: bytes) -> bool:
    """True if doc has more Latin than Cyrillic letters."""
    text = doc.decode("utf-8", errors="surrogateescape")
    cyr = sum(1 for ch in text if "Ѐ" <= ch <= "ӿ")
    lat = sum(1 for ch in text if ch.isalpha() and ch < "ɐ")
    return lat > cyr


# label -> (alt script class, predicate routing a doc there)
SPLIT_SCRIPT: dict[str, tuple[str, Callable[[bytes], bool]]] = {
    "sr": ("srl", latin_majority),
    "uz": ("uzc", lambda doc: not latin_majority(doc)),
    "zh": ("zht", hant_majority),
}
ALT_CLASS = {alt: lang for lang, (alt, _) in SPLIT_SCRIPT.items()}  # alt class -> label


def class_of(lang: str, doc: bytes) -> str:
    """Training class of a doc labelled *lang*: the alt class when its script says so."""
    spec = SPLIT_SCRIPT.get(lang)
    return spec[0] if spec and spec[1](doc) else lang


def walk_corpus(root: str | Path, skip_langs: Collection[str] = ()) -> Iterator[Item]:
    """Yield (domain, lang, path) for docs three levels down, sorted."""
    for domain in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for lang_dir in sorted(p for p in domain.iterdir() if p.is_dir()):
            if lang_dir.name in skip_langs:
                continue
            for doc in sorted(lang_dir.glob("*.txt")):
                if doc.is_file():
                    yield domain.name, lang_dir.name, str(doc)


def read_doc(path: str, cap: int = -1) -> tuple[bytes, str]:
    """(bytes, str) of a doc as the identifier scores it, truncated to cap (<= 0 = no cap)."""
    with open(path, "rb") as f:
        return normalize(f.read(cap if cap > 0 else -1))


def cap_bytes(data: bytes, cap: int = DOC_CAP) -> bytes:
    """Truncate to cap bytes on a codepoint boundary."""
    if len(data) <= cap:
        return data
    while cap > 0 and (data[cap] & 0xC0) == 0x80:
        cap -= 1
    return data[:cap]


def drop(corpus: str | Path, paths: Iterable[str]) -> None:
    """Move docs to a sibling <corpus>_dropped tree, keeping relative paths."""
    dropped_root = Path(str(corpus).rstrip("/") + "_dropped")
    for p in paths:
        src = Path(p)
        dst = dropped_root / src.relative_to(corpus)
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dst)


def pmap_chunks(
    fn: Callable[..., R], tasks: Sequence[Any], jobs: int = 1, shared: tuple[Any, ...] = ()
) -> Iterator[R]:
    """Yield fn(*shared, chunk) over one contiguous chunk of tasks per job."""
    size = max(1, -(-len(tasks) // max(1, jobs)))
    with MapPool(jobs) as f:
        yield from f(
            partial(fn, *shared), [tasks[i : i + size] for i in range(0, len(tasks), size)]
        )


@contextmanager
def MapPool(processes: int = 1) -> Iterator[Callable[..., Iterator[Any]]]:
    """Process pool that falls back to serial map when processes <= 1."""
    if processes > 1:
        ctx = mp.get_context("fork" if sys.platform == "darwin" else None)
        with ctx.Pool(processes) as pool:
            yield pool.imap_unordered
    else:
        yield map
