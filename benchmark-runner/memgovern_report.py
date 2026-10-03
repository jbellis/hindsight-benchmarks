"""Render the complete, independently checked MemGovern comparison."""

import numpy as np

from memgovern_experiment import OUTPUT, PATHS
from memgovern_prepare import MODEL_NAMES
from rerank_eval import ROOT, read_json

NAMES = {
    "bge-small": "BGE-small (Hindsight default)",
    "granite": "Granite Small English R2",
    "leaf": "MongoDB Leaf IR",
    "voyage-1024": "Voyage 4 Nano (1024d)",
}
LABELS = {
    "dense": "Dense only",
    "jev-listwise": "Hybrid + Jev listwise",
    "ettin-150m": "Hybrid + Ettin 150M",
}


def absolute_interval(summary, metric):
    values = np.array(
        [row[metric] for _, row in sorted(summary["per_repository"].items())]
    )
    rng = np.random.default_rng(20261003)
    samples = rng.integers(0, len(values), size=(5000, len(values)))
    return np.quantile(values[samples].mean(axis=1), [0.025, 0.975])


def report():
    result = read_json(OUTPUT / "summary.json")
    if len(result["results"]) != 12 or result["independent_metric_checks"] != 691200:
        raise ValueError("Expected all twelve validated 9600-query combinations")
    lines = [
        "# MemGovern embeddings and rerankers",
        "",
        "All twelve combinations evaluate the same **9,600 queries: 200 per repository across 48 repositories**, selected by the lowest SHA256 of seed20261003, repository and original query ID. Each query searches its repository’s **entire corpus**, covering 121,475 experience cards in total. Candidate selection never uses relevance labels; no gold card is inserted into a candidate pool.",
        "",
        "The primary numbers are repository-macro averages. Query-macro averages are identical for this balanced sample. Dense-only uses normalized embedding cosine top100. Both rerankers receive the same top100 pool from full-corpus BM25/dense RRF, with k=60 and one-based ranks. Jev uses the vendored Hindsight listwise Choice algorithm with pruning off; Ettin uses independent query/document scores.",
        "",
        "| Embedding | Retrieval path | nDCG@10 (95% CI) | MRR@100 | Recall@5 | Recall@100 | Successful Jev API cost |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for embedding in MODEL_NAMES:
        for path in PATHS:
            summary = result["results"][embedding + "/" + path]
            m = summary["repository_macro"]
            lo, hi = absolute_interval(summary, "ndcg_at_10")
            cost = (
                f"${summary['usd_successful_requests']:.3f}"
                if path == "jev-listwise"
                else "Local compute unpriced"
            )
            lines.append(
                f"| {NAMES[embedding]} | {LABELS[path]} | {m['ndcg_at_10']:.4f} [{lo:.4f}, {hi:.4f}] | {m['mrr']:.4f} | {m['recall_at_5']:.4f} | {m['recall_at_100']:.4f} | {cost} |"
            )
    lines.extend(
        [
            "",
            "## Paired differences versus BGE-small",
            "",
            "Differences are nDCG@10, with 5,000 paired bootstrap samples of whole repositories. Intervals are 95% percentile intervals, unadjusted for multiple comparisons. They describe uncertainty across repositories for this fixed query sample, not the additional variation from drawing another sample of queries.",
            "",
            "| Embedding minus BGE-small | Retrieval path | nDCG@10 difference (95% CI) |",
            "|---|---|---:|",
        ]
    )
    for pair in result["paired_comparisons"]:
        if pair["a"] != "bge-small":
            continue
        delta = pair["ndcg_at_10"]
        # Stored comparisons are a minus b; present new model minus baseline.
        lo, hi = [-delta["ci95"][1], -delta["ci95"][0]]
        lines.append(
            f"| {NAMES[pair['b']]} | {LABELS[pair['path']]} | {-delta['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}] |"
        )
    total_cost = sum(
        row["usd_successful_requests"] for row in result["results"].values()
    )
    lines.extend(
        [
            "",
            "## What these labels establish",
            "",
            "MemGovern’s public qrels label one source experience card per original bug query. The task measures recovery of the card distilled from that bug’s fix. Other useful or semantically equivalent cards receive no credit. This does not directly measure whether retrieved experiences help solve a different bug, nor agent task success. Models may have encountered the underlying public repositories or benchmark data during training; this experiment does not test contamination.",
            "",
            "The source is [KaLM-Embedding/LMEB](https://huggingface.co/datasets/KaLM-Embedding/LMEB), pinned to `f9d4294be4a24b8c16d6bfb59d56a6fec4bd95c0`. The official per-repository corpus boundaries and positive associations are retained. Queries are sampled independently of text, labels and model scores, without filtering hard or unsuccessful queries. [Source hashes and counts](../results/experiments/memgovern/source.json).",
            "",
            "## Models and execution",
            "",
            "BGE-small uses 384 dimensions and empty query/document prompts, matching the current Hindsight default. Granite uses 384 dimensions, CLS pooling and empty prompts. Leaf uses its full native 768 dimensions and documented retrieval query prefix; documents are unprefixed. Voyage Nano uses its documented distinct query/document prompts and truncates to 1024 dimensions before normalization. All embeddings use float32 SDPA and unit normalization. All model revisions and prompts are recorded in each frozen pool’s encoding metadata; none are quantized.",
            "",
            "Nano runs on the RTX PRO 5000 Blackwell first. The CPU models run concurrently, each with eight worker processes and four Torch threads per worker. Ettin then owns the GPU, using pinned revision `025501c4e0f9bbeb4c5b198318e0089ff061cc14`, BF16 weights, batch32, native maximum context8192 and Identity activation. Hosted Jev1.13 runs concurrently with eight independent query workers, each paced at two requests per second. Concurrent timings are not isolated speed benchmarks. Leaf’s initial worker pool failed before producing a checkpoint; its identical restart supplied the measured vectors.",
            "",
            f"Total successful Jev input-token cost across four embeddings is **${total_cost:.3f}**, at $0.042 per million input tokens. Output tokens are uncharged. Costs exclude warmup and unknown billing for failed requests/retries; local electricity and hardware are unpriced. Raw successful usage and retry counts are retained. Jev’s listwise algorithm may split oversized pools and use a final selection round; candidate pruning remains off and all100 candidates remain in the final ranking.",
            "",
            "## Validation and reproduction",
            "",
            "The exporter requires all 9,600 queries in all twelve combinations, exact source/pool/run provenance, finite scores and complete candidate permutations. It independently recomputes nDCG@10, MRR@100 and Recall@1/5/10/100 with pytrec_eval: **691,200 metric comparisons**. Retrieval misses count as zero; incomplete runs produce no completed summary. [Summaries, paired intervals, frozen pools and full per-query records](../results/experiments/memgovern/summary.json).",
            "",
            "Download `Procedural/MemGovern/<repository>/{corpus.jsonl,queries.jsonl,qrels.tsv}` from the pinned dataset revision into `<data>/Procedural/MemGovern/`, and put the revision in `<data>/revision.txt`. Use the package versions recorded in pool metadata and cached pinned models. From the repository root:",
            "",
            "    python benchmark-runner/memgovern_prepare.py --embedding voyage-1024 --device cuda:0 --threads 8 --data <data> --work <work>",
            "    python benchmark-runner/memgovern_prepare.py --embedding bge-small --data <data> --work <work>",
            "    python benchmark-runner/memgovern_prepare.py --embedding granite --data <data> --work <work>",
            "    python benchmark-runner/memgovern_prepare.py --embedding leaf --data <data> --work <work>",
            "    python benchmark-runner/memgovern_experiment.py freeze --data <data> --work <work>",
            "    python benchmark-runner/memgovern_experiment.py rerank --reranker jev-listwise --data <data> --work <work>",
            "    python benchmark-runner/memgovern_experiment.py rerank --reranker ettin-150m --data <data> --work <work>",
            "    python benchmark-runner/memgovern_experiment.py export --data <data> --work <work>",
            "    python benchmark-runner/memgovern_report.py",
            "",
            "For CPU preparation, set CUDA_VISIBLE_DEVICES to empty, OMP_NUM_THREADS=4 and OPENBLAS_NUM_THREADS=1. The three CPU commands can run concurrently; preparation checkpoints are reused only when their exact source/model/configuration fingerprints match. Model files are loaded from the local cache. For Nano, use the existing isolated Transformers4.57.6 compatibility overlay and the explicit Qwen3Config registration shim; forward computation and weights are unchanged. Other models and rerankers use Transformers5.18.0. Jev reads the authorized key from `~/.secrets/typesafe_api_key`. Source corpora and large embedding arrays are not duplicated into Git; the source manifest, frozen pools and records permit audit/reproduction.",
            "",
            "No production defaults, embedding stores or live installations changed.",
            "",
        ]
    )
    destination = ROOT / "MEMGOVERN_RESULTS.md"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(destination)
    print(f"Rendered {destination}")


if __name__ == "__main__":
    report()
