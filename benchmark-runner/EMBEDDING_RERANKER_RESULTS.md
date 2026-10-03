# LoCoMo embeddings with Jev listwise and Ettin 150M

All six combinations use the same 1,533 LoCoMo questions, source evidence IDs, and complete corpus. Each embedding model supplies dense retrieval; its ranking is fused with unchanged BM25 using 1/(60+rank), then the top 100 candidates are reranked. Gold evidence is never inserted into the candidate pool. These are retrieval-plus-reranking comparisons: the candidate pools differ across embedding models.

| Embedding | Reranker | nDCG@10 (95% CI) | MRR | Recall@5 | Jev API cost, all queries |
|---|---|---:|---:|---:|---:|
| BGE-base-en-v1.5 | Jev 1.13 listwise | 0.802 [0.782, 0.822] | 0.826 | 0.828 | $0.447 |
| BGE-base-en-v1.5 | Ettin 150M | 0.713 [0.695, 0.730] | 0.712 | 0.770 | Local compute unpriced |
| Voyage 4 Nano | Jev 1.13 listwise | 0.839 [0.818, 0.858] | 0.851 | 0.870 | $0.463 |
| Voyage 4 Nano | Ettin 150M | 0.735 [0.718, 0.750] | 0.729 | 0.796 | Local compute unpriced |
| Granite Small English R2 | Jev 1.13 listwise | 0.787 [0.765, 0.810] | 0.812 | 0.811 | $0.437 |
| Granite Small English R2 | Ettin 150M | 0.699 [0.681, 0.719] | 0.701 | 0.756 | Local compute unpriced |

## Retrieval before reranking

Top-100 evidence recall is the mean fraction of annotated evidence turns present in the candidate pool. It is the upper bound on evidence recall for either reranker on that pool.

| Embedding | Unreranked hybrid nDCG@10 | Top-100 evidence recall |
|---|---:|---:|
| BGE-base-en-v1.5 | 0.459 | 0.876 |
| Voyage 4 Nano | 0.539 | 0.933 |
| Granite Small English R2 | 0.445 | 0.852 |

## Paired embedding comparisons

Differences below are nDCG@10, using 5,000 paired bootstrap samples of whole conversations. Intervals are 95% percentile intervals, unadjusted for multiple comparisons. Only ten conversations underlie these estimates.

- jev-listwise, voyage minus BGE: **+0.037 [+0.025, +0.049]**.
- jev-listwise, granite minus BGE: **-0.015 [-0.031, -0.002]**.
- ettin-150m, voyage minus BGE: **+0.022 [+0.015, +0.031]**.
- ettin-150m, granite minus BGE: **-0.013 [-0.023, -0.006]**.

## Models, execution and reproduction

BGE reuses the exact already published candidate fixture and two completed reranker runs, preserving their provenance and sequential timing. No BGE reranker API calls were repeated. Its revision and query prefix remain in [the original fixture](datasets/reranker/locomo10-top100.json.gz).

Voyage uses `voyageai/voyage-4-nano` revision `67fabc9bef010dabc5f6024aa1b1b6b93410426f`, full 2048-dimensional embeddings, and the documented distinct query/document prompts. Granite uses `ibm-granite/granite-embedding-small-english-r2` revision `2ab6fa8ea2d674564defd37171ae19079b864b33`, 384 dimensions and no additional prompt. New embeddings use float32, cosine normalization, batch size 32, SDPA attention and each model’s native maximum context. Preparation is local on the RTX PRO 5000 Blackwell. Official settings: [Voyage model card](https://huggingface.co/voyageai/voyage-4-nano), [Granite model card](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2).

Voyage’s pinned custom code needs the Transformers 4 attention API. Its isolated preparation environment uses Transformers 4.57.6, tokenizers 0.22.2, huggingface-hub 0.36.2 and requests 2.32.5, with the same PyTorch and Sentence Transformers as the existing environment. The adapter explicitly assigns Qwen3Config as the custom model’s config_class for AutoModel registration; weights and forward computation are unchanged. Granite preparation uses Transformers 5.18.0. All core versions and separate preparation hashes are recorded in the fixtures; Granite source is preserved at commit 47534ba, and Voyage/concurrent inference at commit 13d6f2a. Rerankers retain the original environment and models.

Jev is pinned to jev-1.13.0 and uses the existing Hindsight Choice/tournament ranking, pruning disabled. New Jev runs use three concurrent query workers, each owning a separate HTTP client, usage/retry counters, an eight-request concurrency limit and 10 requests/second pacing. One writer records completed queries. Warmup is excluded; interrupted queries cannot silently receive relevance zero. GPU preparation and Ettin work overlap hosted queries. New timings are concurrent execution measurements and cannot be used as a controlled latency comparison with sequential BGE runs; no speed claim is made. Ettin uses the same pinned checkpoint, unquantized bfloat16 and batches of 32 as before.

The Granite Ettin process encountered a CUDA unknown error after 463 queries and resumed from its checkpoint with identical metadata on the same GPU. No query was dropped. The [interruption ledger](../results/experiments/embedding-reranker-locomo/run-interruptions.json) preserves the failure and resolution. Jev monetary values use successful recorded input tokens at the previously verified $0.042/million rate; output is free. Warmup, unknown failed/retried request billing and local compute costs are excluded.

Reproduce preparation with `python benchmark-runner/embedding_experiment.py prepare --embedding voyage` (using the isolated compatibility environment) or `--embedding granite`. Select Blackwell with `CUDA_VISIBLE_DEVICES=GPU-27368065-5d8b-6e5e-4e81-2e924bd9ce73`. Do not overwrite frozen fixtures. Score Jev with `embedding_experiment.py hosted --embedding <name> --workers 3 --records <run-root>/<name> --secrets-dir ~/.secrets`; score Ettin with `run_all_reranker.py run --fixture results/experiments/embedding-reranker-locomo/<name>/fixture.json.gz --model ettin-150m --output <run-root>/<name>`. Export with `embedding_experiment.py export --records <run-root>`. Dependencies are the existing pinned runner environment plus pytrec-eval-terrier 0.5.10 for independent validation.

Validation: **24 focused behavior tests pass**, Ruff and diff checks pass, and **36,816 independent TREC metric comparisons** match all six complete results. [Frozen fixtures, aggregates and raw rankings](../results/experiments/embedding-reranker-locomo/); [all paired comparisons](../results/experiments/embedding-reranker-locomo/paired-comparisons.json).

The original LoCoMo labels remain incomplete, and public benchmark exposure is unknown. This is not a clean holdout or end-to-end Hindsight evaluation. Changing retrieval also changes the candidate order and decoys seen by Jev’s listwise classifier, so differences do not isolate embedding quality from reranker interactions.
