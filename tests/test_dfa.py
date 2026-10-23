import numpy as np
import pytest

from py3langid.dfa import MAX_FEATURE_BYTES, build_dfa, byte_classes, visit_counts

from .conftest import longest_counts


@pytest.mark.parametrize("data", [b"", b"c", b"zzz", b"xy", b"xabcy",
                                 b"abcabc", b"aabc"])
def test_build_dfa_longest_match(data):
    """the DFA counts, over byte positions, the longest feature ending there"""
    feats = [b"ab", b"abc", b"bc", b"c", b"xy", b"aab"]
    assert visit_counts(build_dfa(feats), data) == longest_counts(feats, data)


def test_build_dfa_random():
    rng = np.random.default_rng(0)
    for _ in range(20):
        feats = sorted({bytes(rng.integers(97, 101, rng.integers(1, MAX_FEATURE_BYTES + 1)).tolist())
                        for _ in range(30)})
        dfa = build_dfa(feats)
        for _ in range(5):
            data = bytes(rng.integers(96, 102, 40).tolist())
            assert visit_counts(dfa, data) == longest_counts(feats, data)


def test_build_dfa_full_width_keys():
    """8-byte features over byte classes up to 255 fill every bit of the prefix key"""
    feats = [bytes([b]) for b in range(256)] + [b"\xff" * 8, b"\xfe\xff" * 4, b"\xff" * 7 + b"\xfe"]
    data = b"\xff" * 9 + b"\xfe" + b"\xfe\xff" * 5 + b"\x00\xff"
    assert visit_counts(build_dfa(feats), data) == longest_counts(feats, data)


def test_build_dfa_rejects_long_features():
    with pytest.raises(ValueError, match="longer than"):
        build_dfa([b"a" * (MAX_FEATURE_BYTES + 1)])


def test_byte_classes():
    """bytes used by a feature get one class each, every other byte the root's"""
    table, k = byte_classes([b"ab", b"ba", b"c"])
    assert k == 4 and sorted(set(table)) == [0, 1, 2, 3]
    assert table[ord("a")] != table[ord("b")] and table[0] == table[255] == 3
    assert byte_classes([bytes(range(256))])[1] == 256


def test_build_dfa_shares_every_duplicate_row():
    """row sharing is maximal: no two stored rows hold the same transitions"""
    feats = [b"ab", b"abc", b"bc", b"c", b"xy", b"aab", b"bca", b"cab"]
    dfa = build_dfa(feats)
    k = len(set(dfa.table))
    stored = len(dfa.nextmove) // k
    assert stored < len(dfa.output)  # sharing actually happened
    contents = {tuple(dfa.nextmove[r:r + k]) for r in dfa.rowbase}
    assert len(contents) == stored


def test_build_dfa_uint32_widening():
    """state ids beyond uint16 switch the rows to uint32"""
    alphabet = range(41)
    feats = [bytes([a, b, c]) for a in alphabet for b in alphabet for c in alphabet]
    dfa = build_dfa(feats)
    assert len(dfa.output) > 1 << 16 and dfa.nextmove.typecode == "I"
    assert visit_counts(dfa, bytes([40, 40, 40])) == {len(feats) - 1: 1}
    assert build_dfa([b"ab"]).nextmove.typecode == "H"
