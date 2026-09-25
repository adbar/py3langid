"""Topup: eval sub-corpora never enter training, dataset revisions are pinned."""
import sys
import types

from py3langid.train import topup
from py3langid.train.common import MIN_DOC


def test_glot_docs_skips_eval_sources(monkeypatch):
    calls = []
    keep, flores = "a" * MIN_DOC, "b" * MIN_DOC

    def load_dataset(repo, config, **kw):
        calls.append((repo, config, kw["revision"]))
        return [{"text": keep, "dataset": "Wikipedia"}, {"text": flores, "dataset": "Flores200"}]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    docs = list(topup._glot_docs(topup.GLOT500_REPO, "text", {"kik": {"kik_Latn"}}, "kik"))
    assert docs == [keep.encode()]
    assert calls == [(topup.GLOT500_REPO, "kik_Latn", topup.REVISION[topup.GLOT500_REPO])]


def test_repo_configs_pinned(monkeypatch):
    seen = []

    def list_repo_files(repo, repo_type, revision):
        seen.append(revision)
        return ["kik_Latn/train.parquet", "README.md", "data/sot_Latn/x.parquet"]

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(list_repo_files=list_repo_files))
    assert topup._repo_configs(topup.GLOTCC_REPO) == {"kik": {"kik_Latn"}, "sot": {"sot_Latn"}}
    assert seen == [topup.REVISION[topup.GLOTCC_REPO]]


def test_needy_counts_domains_with_enough_docs(tmp_path):
    def cell(domain, cls, n):
        (tmp_path / domain / cls).mkdir(parents=True)
        for i in range(n):
            (tmp_path / domain / cls / f"doc{i:04d}.txt").write_bytes(b"x")

    for d in ("wiki", "cc100", "leipzig"):
        cell(d, "de", 50)
    cell("wiki", "arz", 300)
    cell("tatoeba", "arz", 31)
    cell("leipzig", "arz", 300)
    assert topup.needy(tmp_path, ["de", "arz", "pcm", "om"]) == ["arz", "om"]
