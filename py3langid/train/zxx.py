"""Synthetic not-a-language (zxx) training docs. Deterministic per-domain seeds."""
import base64
import json
import random
import string
from pathlib import Path

from .common import cap_bytes

DOCS_PER_DOMAIN = 300
DOMAIN_SEEDS = {"wiki": 1, "cc100": 2}

GENRES = (
    lambda rng, n: " ".join(str(rng.randint(0, 10**rng.randint(1, 9))) for _ in range(8)),
    lambda rng, n: "".join(rng.choice("!@#$%^&*()_+-=[]{};:,.<>/?|~`\"'\\") for _ in range(n)),
    lambda rng, n: "".join(rng.choice(string.ascii_letters + " ") for _ in range(n)),
    lambda rng, n: "".join(chr(rng.choice([rng.randint(0x2200, 0x23FF), rng.randint(0x2500, 0x27BF), rng.randint(0x1F300, 0x1F5FF)])) for _ in range(n // 2)),
    lambda rng, n: base64.b64encode(rng.randbytes(n)).decode(),
    lambda rng, n: rng.randbytes(n // 2).hex(),
    lambda rng, n: " ".join(f"https://ex{rng.randint(1,999)}.com/{rng.randbytes(4).hex()}?id={rng.randint(1,9999)}" for _ in range(3)),
    lambda rng, n: "".join(f"<t{rng.randint(1,99)} a='{rng.randbytes(3).hex()}'/>" for _ in range(6)),
    lambda rng, n: json.dumps({f"k{rng.randint(1,99)}": rng.randint(0, 9999) for _ in range(5)}),
    lambda rng, n: " ".join(["".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(2, 6)))] * rng.randint(5, 15)),
)


def _doc(rng):
    genre = GENRES[rng.randrange(len(GENRES))]
    lines = [genre(rng, rng.randint(20, 70)) for _ in range(rng.randint(30, 60))]
    return cap_bytes("\n".join(lines).encode())


def ensure_zxx(corpus):
    """Write zxx dirs if absent. Returns docs written."""
    written = 0
    for domain, seed in DOMAIN_SEEDS.items():
        out = Path(corpus) / domain / "zxx"
        if out.is_dir() and len(list(out.glob("*.txt"))) >= DOCS_PER_DOMAIN:
            continue
        out.mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed)
        for i in range(DOCS_PER_DOMAIN):
            (out / f"doc{i:04d}.txt").write_bytes(_doc(rng))
            written += 1
    return written

