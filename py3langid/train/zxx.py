"""Synthetic not-a-language (zxx) training docs: numbers, symbols,
gibberish, base64/hex, URLs, markup, config soup, repeated tokens.
Deliberately NO real code (user decision 2026-08-26: code reads as en).
Deterministic seeds per domain reproduce the validated class exactly.

    python -m py3langid.train.zxx CORPUS_DIR
"""
import base64
import json
import random
import string
import sys
from pathlib import Path

DOCS_PER_DOMAIN = 150
DOMAIN_SEEDS = {"wiki": 1, "cc100": 2}

def _doc(rng):
    genre = rng.randrange(10)
    lines = []
    for _ in range(rng.randint(30, 60)):
        n = rng.randint(20, 70)
        if genre == 0:
            lines.append(" ".join(str(rng.randint(0, 10**rng.randint(1, 9))) for _ in range(8)))
        elif genre == 1:
            lines.append("".join(rng.choice("!@#$%^&*()_+-=[]{};:,.<>/?|~`\"'\\") for _ in range(n)))
        elif genre == 2:
            lines.append("".join(rng.choice(string.ascii_letters + " ") for _ in range(n)))
        elif genre == 3:
            lines.append("".join(chr(rng.choice([rng.randint(0x2200, 0x23FF), rng.randint(0x2500, 0x27BF), rng.randint(0x1F300, 0x1F5FF)])) for _ in range(n // 2)))
        elif genre == 4:
            lines.append(base64.b64encode(rng.randbytes(n)).decode())
        elif genre == 5:
            lines.append(rng.randbytes(n // 2).hex())
        elif genre == 6:
            lines.append(" ".join(f"https://ex{rng.randint(1,999)}.com/{rng.randbytes(4).hex()}?id={rng.randint(1,9999)}" for _ in range(3)))
        elif genre == 7:
            lines.append("".join(f"<t{rng.randint(1,99)} a='{rng.randbytes(3).hex()}'/>" for _ in range(6)))
        elif genre == 8:
            lines.append(json.dumps({f"k{rng.randint(1,99)}": rng.randint(0, 9999) for _ in range(5)}))
        else:
            tok = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(2, 6)))
            lines.append(" ".join([tok] * rng.randint(5, 15)))
    return "\n".join(lines).encode()[:3000]


def ensure_zxx(corpus):
    """Write the zxx dirs if absent; returns number of docs written."""
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


if __name__ == "__main__":
    print("zxx docs written:", ensure_zxx(sys.argv[1]))
