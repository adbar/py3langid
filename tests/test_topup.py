"""Topup: eval sub-corpora never enter training, dataset revisions are pinned."""
import sys
import types

from py3langid.train import topup
from py3langid.train.common import MIN_DOC

from .conftest import write_corpus


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


def test_glot_config_picks_the_class_script():
    configs = {"cmn": {"cmn_Hans", "cmn_Hant"}, "uzn": {"uzn_Latn", "uzn_Cyrl"},
               "deu": {"deu_Latn"}, "srp": {"srp_Cyrl", "srp_Latn"}, "fra": {"fra_Latn", "fra_Brai"}}
    assert [topup._glot_config(configs, c) for c in ("zh", "zht", "uz", "uzc", "srl", "de")] == [
        "cmn_Hans", "cmn_Hant", "uzn_Latn", "uzn_Cyrl", "srp_Latn", "deu_Latn"]
    assert topup._glot_config(configs, "fr") is None  # ambiguous, no script preference
    assert topup._glot_config(configs, "om") is None


def test_needy_counts_domains_with_enough_docs(tmp_path):
    cells = [("wiki", "de", 50), ("cc100", "de", 50), ("leipzig", "de", 50),
             ("wiki", "arz", 300), ("tatoeba", "arz", 31), ("leipzig", "arz", 300)]
    write_corpus(tmp_path, [(d, c, b"x") for d, c, n in cells for _ in range(n)])
    assert topup.needy(tmp_path, ["de", "arz", "pcm", "om"]) == ["arz", "om"]


def test_topup_retries_failed_stream_only(monkeypatch, tmp_path):
    streamed = []
    fail = {"om"}

    def glot_docs(repo, field, configs, cls):
        streamed.append(cls)
        for i in range(3):
            yield f"{cls} doc {i} ".encode() * MIN_DOC
        if cls in fail:
            raise OSError("stream dropped")

    monkeypatch.setattr(topup, "_repo_configs", lambda repo: {})
    monkeypatch.setattr(topup, "_glot_docs", glot_docs)
    topup.gather_topup(tmp_path, ["om", "so"], max_docs=5, jobs=1)
    assert (tmp_path / "glotcc" / "om" / topup.PARTIAL).exists()
    fail.clear()
    streamed.clear()
    topup.gather_topup(tmp_path, ["om", "so"], max_docs=5, jobs=1)
    assert streamed == ["om"]
    assert not (tmp_path / "glotcc" / "om" / topup.PARTIAL).exists()
    assert len(list((tmp_path / "glotcc" / "om").glob("*.txt"))) == 3
