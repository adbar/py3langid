"""Clean a gathered corpus: zxx, dedup, class rules."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from .dedup import dedup
from .rules import apply_rules
from .zxx import ensure_zxx


def clean(corpus: str | Path) -> None:
    print("zxx docs written:", ensure_zxx(corpus))
    print("dedup: removed", dedup(corpus), "duplicate lines")
    dropped, stripped = apply_rules(corpus)
    print("rules: dropped docs", dropped, "stripped bytes", stripped)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", metavar="CORPUS_DIR")
    clean(parser.parse_args(argv).corpus)


if __name__ == "__main__":
    main()
