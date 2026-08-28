"""Unit tests for split-script routing and the self-verify decision."""
from py3langid.train.common import latin_majority, route_script
from py3langid.train.gather_data import MIN_DOC, write_docs
from py3langid.train.verify import is_foreign


def test_latin_majority():
    assert latin_majority("Republika Srbija je država".encode())
    assert not latin_majority("Република Србија је држава".encode())
    assert not latin_majority(b"12345 ...")  # no letters -> not Latin-majority


def test_route_script(tmp_path):
    sr = tmp_path / "sr"
    assert route_script(sr, "Република".encode()) == sr
    assert route_script(sr, b"Republika") == tmp_path / "srl"
    other = tmp_path / "hr"
    assert route_script(other, b"Republika") == other


def test_write_docs_splits_sr(tmp_path):
    cyr = "Београд је главни град Србије. ".encode() * 30
    lat = b"Beograd je glavni grad Srbije. " * 30
    assert len(cyr) >= MIN_DOC and len(lat) >= MIN_DOC
    n = write_docs(tmp_path / "sr", [cyr, lat, cyr, lat], max_docs=10)
    assert n == 4
    assert len(list((tmp_path / "sr").iterdir())) == 2
    assert len(list((tmp_path / "srl").iterdir())) == 2
    assert (tmp_path / "srl" / "doc0000.txt").read_bytes() == lat.strip()[:3000]


def test_is_foreign():
    assert is_foreign("xh", "en")
    assert not is_foreign("xh", "zu")       # confusable pair protected
    assert not is_foreign("sr", "mk")       # protect-list regression guard
    assert not is_foreign("srl", "hr")      # alias srl->sr, sr/hr protected
    assert not is_foreign("no", "nb")       # legacy nb aliases to no
    assert is_foreign("az", "la")
    assert not is_foreign("de", "de")
