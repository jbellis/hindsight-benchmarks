import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rerank_eval import (
    Remote,
    confidence,
    digest,
    evidence_ids,
    load_records,
    locomo_source,
    mapped_scores,
    metrics,
    rank_scores,
    validate_fixture,
)


def test_recall_counts_all_evidence_instead_of_hit_rate():
    values = metrics(["b", "x", "y"], {"a": 1, "b": 1})
    assert values["recall_at_5"] == 0.5
    assert values["hit_at_5"] == 1
    assert values["ndcg_at_10"] < 1


def test_retrieval_miss_counts_as_zero():
    assert all(v == 0 for v in metrics(["x", "y"], {"a": 1}).values())


def test_rank_matches_original_indices_and_breaks_ties_by_input_order():
    scores = mapped_scores(
        [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.2},
            {"index": 1, "relevance_score": 0.9},
        ],
        3,
    )
    assert rank_scores(scores, 3) == [1, 2, 0]


@pytest.mark.parametrize(
    "rows",
    [
        [{"index": 0, "relevance_score": 0.5}],
        [{"index": 0, "relevance_score": 0.5}, {"index": 0, "relevance_score": 0.7}],
        [{"index": -1, "relevance_score": 0.5}, {"index": 1, "relevance_score": 0.7}],
        [
            {"index": 0, "relevance_score": float("nan")},
            {"index": 1, "relevance_score": 0.7},
        ],
    ],
)
def test_invalid_provider_results_fail_visibly(rows):
    with pytest.raises(ValueError):
        mapped_scores(rows, 2)


def test_evidence_formatting_is_normalized_without_guessing():
    assert evidence_ids(["D8:6; D9:17", "D:11:26", "D30:05"]) == [
        "D8:6",
        "D9:17",
        "D11:26",
        "D30:5",
    ]
    with pytest.raises(ValueError):
        evidence_ids(["D1:18", "D", "D1:20"])


def test_dataset_exclusions_are_independent_of_retrieval(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "chat",
                    "conversation": {
                        "session_1_date_time": "today",
                        "session_1": [
                            {"dia_id": "D1:1", "speaker": "A", "text": "Evidence"}
                        ],
                    },
                    "qa": [
                        {"question": "valid", "category": 1, "evidence": ["D1:1"]},
                        {"question": "broken", "category": 1, "evidence": ["D1:9"]},
                        {"question": "empty", "category": 1, "evidence": []},
                    ],
                }
            ]
        )
    )
    _, questions, excluded, _ = locomo_source(path)
    assert [q["query"] for q in questions] == ["valid"]
    assert len(excluded) == 2


def test_fixture_tampering_is_rejected():
    fixture = {
        "corpus": {"a": "A"},
        "queries": [{"id": "q", "relevant": {"a": 1}, "candidates": ["a"]}],
    }
    fixture["fixture_sha256"] = digest(fixture)
    validate_fixture(fixture)
    fixture["corpus"]["a"] = "changed"
    with pytest.raises(ValueError, match="hash"):
        validate_fixture(fixture)


def test_resume_rejects_changed_adapter_or_fixture(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps({"run_sha256": "old", "query_id": "q"}) + "\n")
    with pytest.raises(ValueError, match="Resume refused"):
        load_records(path, "new")


def test_duplicate_resume_records_are_rejected(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text((json.dumps({"run_sha256": "same", "query_id": "q"}) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        load_records(path, "same")


def test_jev_receives_only_one_candidate_in_each_state():
    remote = object.__new__(Remote)
    remote.config = {"provider": "typesafe", "model": "jev-1.13.0"}
    bodies = []

    async def post(body):
        bodies.append(body)
        candidate = body["state"]["candidate"]
        return {
            "model": "jev-1.13.0",
            "answers": {"relevance": {"noul": 0.9 if candidate == "gold" else 0.1}},
            "usage": {"input_tokens": 10},
        }, 0.01

    remote.post = post
    scores, usage, _ = asyncio.run(remote.score("query", ["gold", "other"]))
    assert scores == [0.9, 0.1]
    assert usage["input_tokens"] == 20
    assert [body["state"] for body in bodies] == [
        {"query": "query", "candidate": "gold"},
        {"query": "query", "candidate": "other"},
    ]


def test_bootstrap_preserves_conversation_clustering():
    rows = [{"group": "a", "metrics": {"mrr": 1.0}}] * 100 + [
        {"group": "b", "metrics": {"mrr": 0.0}}
    ] * 100
    assert confidence(rows, "mrr", "locomo") == [0.0, 1.0]


def test_paired_interval_compares_same_queries_and_cancels_shared_variation():
    from summarize_rerankers import paired_interval

    a = {
        "q1": {"group": "chat1", "metrics": {"mrr": 0.8}},
        "q2": {"group": "chat2", "metrics": {"mrr": 0.4}},
    }
    b = {
        "q1": {"group": "chat1", "metrics": {"mrr": 0.6}},
        "q2": {"group": "chat2", "metrics": {"mrr": 0.2}},
    }
    result = paired_interval(a, b, "mrr")
    assert result["delta"] == pytest.approx(0.2)
    assert result["ci95"] == pytest.approx([0.2, 0.2])
    with pytest.raises(ValueError, match="identical queries"):
        paired_interval(a, {"q1": b["q1"]}, "mrr")


def test_partial_runs_cannot_publish_quality(tmp_path):
    from types import SimpleNamespace
    from rerank_eval import summarize, write_json

    fixture = {
        "schema_version": 2,
        "dataset": "scifact",
        "corpus": {"a": "Evidence", "b": "Other"},
        "queries": [
            {
                "id": "q1",
                "group": "q1",
                "category": "fact-checking",
                "query": "First",
                "relevant": {"a": 1},
                "candidates": ["a", "b"],
            },
            {
                "id": "q2",
                "group": "q2",
                "category": "fact-checking",
                "query": "Second",
                "relevant": {"a": 1},
                "candidates": ["a", "b"],
            },
        ],
    }
    fixture["fixture_sha256"] = digest(fixture)
    fixture_path = tmp_path / "fixture.json.gz"
    write_json(fixture_path, fixture)
    metadata = {
        "config": {"provider": "rrf", "pricing_type": "free"},
        "fixture_sha256": fixture["fixture_sha256"],
    }
    path = tmp_path / "records" / "scifact" / "rrf.jsonl"
    path.parent.mkdir(parents=True)
    write_json(path.with_suffix(".metadata.json"), metadata)
    path.write_text(
        json.dumps({"run_sha256": digest(metadata), "query_id": "q1"}) + "\n"
    )
    output = tmp_path / "published"
    with pytest.raises(ValueError, match="Incomplete run"):
        summarize(
            SimpleNamespace(
                fixture=fixture_path,
                model="rrf",
                records=tmp_path / "records",
                output=output,
            )
        )
    assert not output.exists()
