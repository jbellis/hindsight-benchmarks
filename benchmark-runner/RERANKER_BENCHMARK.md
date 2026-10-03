# Direct reranker benchmark

This benchmark measures passage ordering on identical, frozen candidate lists. Each model receives exactly the same query and candidate text. It reports evidence retrieval quality, uncertainty, client-observed latency, retries, and API usage separately. It does not measure Hindsight's extraction, graph retrieval, observation consolidation, or answer generation.

## Why the previous results were removed

The March 3, 2026 results measured an unpinned Hindsight `recall()` pipeline, rather than the reranker alone. Historical code applied sigmoid to every provider's scores, including remote scores already bounded by zero and one. The March implementation then sorted by `0.6 * sigmoid(score) + 0.2 * normalized_RRF + 0.1 * temporal + 0.1 * recency`. This compressed a remote model's entire relevance contribution into a range approximately 0.139 wide, while the upstream RRF contribution alone could span 0.2. A local model producing logits had a substantially different effective influence. The score-handling bug was fixed on May 25, 2026; the old result files were never updated. Their exact runtime image is not recorded, so they should not be used to infer a fair model ranking.

The old annotations also came from Gemini 2.5 Flash judging a retrieved subset of one conversation's extracted facts. Questions with no selected facts were excluded. The old `Recall@K` was actually an any-hit rate. The replacement removes those annotations, aggregates, and unsupported explanations about commercial models being poorly suited to the domain.

References: [original benchmark commit](https://github.com/vectorize-io/hindsight-benchmarks/commit/32bfde9384703224c02c175594ed665c28201870), [historical normalization](https://github.com/vectorize-io/hindsight/blob/719e79a4d9ada787fa1899742d1b52cd89ed1dee/hindsight-api/hindsight_api/engine/search/reranking.py#L90), [historical combined scoring](https://github.com/vectorize-io/hindsight/blob/719e79a4d9ada787fa1899742d1b52cd89ed1dee/hindsight-api/hindsight_api/engine/memory_engine.py#L2590), [normalization fix](https://github.com/vectorize-io/hindsight/pull/1512).

## Datasets and labels

**LoCoMo:** all ten conversations from the original `snap-research/locomo` release. A document is one dialogue turn, with its speaker, session timestamp, and any provided image caption/search description. We use the dataset's original evidence IDs, not model-generated fact annotations. There are 1,986 source questions: 446 adversarial questions are excluded, four non-adversarial questions have no evidence annotations, and three have unresolved evidence references. The resulting 1,533 questions are fixed for every model. Syntax normalization splits packed ID lists, accepts the extra colon in `D:11:26`, and removes leading zeros from numeric ID components. It never guesses a nonexistent turn. Every correction and exclusion is saved in the fixture. The evidence sets are binary and can be incomplete: an unannotated useful passage may receive no credit. This is retrieval of annotated dialogue evidence, not a test of extracted memory facts. A diagnostic inspection found concrete incomplete labels: for conv-48/q41 (ways of enhancing yoga practice), the top Voyage passage explicitly mentions candles and essential oils but is absent from the source evidence set; for conv-44/q58 (dinner on October 24), a passage identifying sushi for that evening is similarly unlabelled. These examples were selected from top-1 misses and are not a random estimate of label error frequency. We leave source labels fixed and publish this limitation rather than changing labels after seeing model outputs.

**SciFact:** the BEIR test split, 300 queries against 5,183 scientific abstracts. Documents contain title and abstract. Positive qrels identify evidence useful for supporting or contradicting a claim; both count as relevant. This supplies an independent domain with expert-written claims and annotated scientific evidence, but is still a public benchmark.

The frozen files are `datasets/reranker/locomo10-top100.json.gz` and `datasets/reranker/scifact-test-top100.json.gz`. Each includes the corpus, queries, evidence labels, candidate IDs in input order, preparation parameters, source checksum, and a verified content hash. Expected answers, summaries, observations, and gold IDs are never passed to a reranker.

Sources: [LoCoMo dataset](https://github.com/snap-research/locomo), [LoCoMo paper](https://arxiv.org/abs/2402.17753), [BEIR dataset download](https://github.com/beir-cellar/beir/wiki/Datasets-available), [SciFact paper](https://arxiv.org/abs/2004.14974).

## Training exposure and interpretation

These are **public-benchmark results, not a claim of unseen-data or zero-shot performance**. Training exposure for the hosted models and their base models is not independently known. Corpus familiarity alone is weaker evidence of contamination than exposure to query/evidence pairs or repeated model selection against their labels.

Ettin explicitly monitored NanoBEIR throughout training, selected checkpoints using it, and selected the released checkpoint using full MTEB retrieval scores. NanoBEIR includes SciFact. This is disclosed benchmark-driven model selection, and means SciFact cannot be treated as a clean holdout for Ettin. Hindsight's Jev integration was itself developed using LoCoMo experiments; our independently specified pointwise Jev adapter is evaluated without tuning the rubric against these results. A new private holdout is required before claiming generalization to fresh tasks or choosing a production model solely on these rankings. [Ettin training and model selection](https://huggingface.co/blog/ettin-reranker#evaluation).

## Frozen retrieval

BM25 (`rank-bm25`, default parameters, Unicode word tokenization) and normalized embeddings from a pinned `BAAI/bge-base-en-v1.5` revision independently rank the entire applicable corpus. LoCoMo searches within the question's conversation; SciFact searches all abstracts. The dense query prefix is `Represent this sentence for searching relevant passages: `; documents have no prefix. Both ranks start at one and are fused as `1/(60 + lexical_rank) + 1/(60 + dense_rank)`. All corpus documents participate. The top 100 are frozen once. Stable ties use lexicographically sorted document IDs.

Gold evidence is never injected into candidates. Questions whose evidence is not retrieved remain in the denominator and score zero. Summaries report the candidate evidence recall ceiling so retrieval limitations remain visible. There is no per-model ingestion, retrieval, label creation, or question filtering.

## Model scoring and latency

MiniLM L6/L12, Ettin 150M/400M, and Qwen3-Reranker-0.6B run through Sentence Transformers on the same RTX PRO 5000 Blackwell GPU. Weights and compute are unquantized bfloat16. Each model receives three warmup batches. The batch size is 32. The recorded maximum input lengths are 512 for MiniLM and 8,192 for Ettin/Qwen; any truncated local pairs are counted. We use identity score activation and sort the original model score, without cross-provider normalization, thresholds, or recency blending. Each run records the model revision, CUDA version, precision, device, library versions, and adapter source hash. These latency measurements characterize this configuration, not each model's maximum throughput.

Voyage uses the documented MongoDB endpoint for the supplied Atlas model key. For a legacy Voyage key, explicitly change the configured endpoint before running; the harness does not silently try a second endpoint. Voyage truncation is disabled. Cohere uses the v2 rerank API. Both adapters require exactly one finite score at each original candidate index; missing, duplicate, or out-of-range indices fail visibly.

Jev is pinned to `jev-1.13.0` and evaluated **pointwise**: one separate Noul request per `(query, candidate)` pair. Every request has only that query and that candidate in `state`. There is no shared pool, Choice question, tournament, candidate filtering, or cross-pool probability comparison. The fixed question asks whether the candidate supplies evidence useful for answering or verifying the query, explicitly counting contradiction and partial multi-fact evidence. The full rubric is recorded in metadata. Each candidate's actual reported input/output usage is accumulated. The run uses 64 simultaneous requests at most, paced to 60 requests/second; the other hosted runs are paced to at most 10 requests/second.

Warm latency starts immediately before scoring a query's whole candidate pool and includes local token-length auditing, client pacing/queueing, retries, and network round trips. It excludes fixture loading, candidate retrieval, model download, initialization, and warmup. Jev's request-level latency is not presented as its whole-query latency. Local GPU jobs run sequentially; independent hosted jobs run concurrently. We publish p50/p95 and mean, not just the successful final attempt's time.

Transient HTTP 429/5xx and transport failures retry with bounded backoff. Retry counts are included. A query that still fails stops the run and creates an explicit local failure report; it is never scored as an irrelevant result. Resume requires the same fixture, adapter, model configuration, environment, hardware, and concurrency. A complete quality result is emitted only after every fixed eligible query succeeds and all stored orderings/metrics are revalidated. No partial run enters the leaderboard.

## Metrics and uncertainty

`nDCG@10` uses gain `2^relevance - 1` and logarithmic rank discount. Its ideal ranking uses all source evidence, including evidence missing from candidates. `MRR` averages the reciprocal rank of the first annotated relevant passage across all fixed queries. `Recall@K` averages `number of relevant passages in top K / total relevant passages` per query. `Hit@K` separately reports whether any annotated evidence is in the first K. This distinction matters for questions requiring multiple passages.

Quality estimates use the arithmetic mean over fixed queries, with categories and individual conversation/query groups also available. The 95% percentile intervals use 5,000 bootstrap samples: whole conversations for LoCoMo, individual queries for SciFact. Whole-conversation resampling preserves dependence between questions from the same conversation. LoCoMo has only ten independent conversations, so its intervals are necessarily coarse. Bootstrap intervals do not capture label errors, training exposure, or sensitivity to retriever choice.

`results/leaderboard/reranker-analysis/paired-comparisons.json` contains paired differences for every model pair on nDCG@10, MRR, and recall@5 using the same sampled groups. These intervals are unadjusted for multiple comparisons; they are descriptive and should not be used to cherry-pick significance.

Cost uses API-reported tokens/search units and verified list rates, excluding credits, warmup, and unknown billing for failed attempts. Voyage 3 is $0.050/million tokens, Voyage 3 Lite $0.020/million tokens, and Jev $0.042/million input tokens with free output. Cohere search-unit counts are recorded but its monetary rate remains unpriced until verified for the applicable contract. Local compute/electricity is unpriced, rather than assigned an invented zero cost. There is no weighted quality/speed/cost leaderboard score.

## Reproduce

The standalone runner has a locked dependency graph and avoids the parent project's machine-specific Hindsight package path. Python 3.12 and a supported CUDA GPU are required for local runs. From `benchmark-runner`:

```sh
uv run --no-project --script run_all_reranker.py run \
  --fixture datasets/reranker/locomo10-top100.json.gz \
  --model voyage-rerank-3 --output reranker-runs --secrets-dir "$HOME/.secrets" \
  --requests-per-second 10
```

Alternatively set `VOYAGE_API_KEY`, `COHERE_API_KEY`, and `TYPESAFE_API_KEY` in the environment. `--secrets-dir` reads files named `voyage_api_key`, `cohere_api_key`, and `typesafe_api_key`. Their contents are used only in Authorization headers and are never written to records. Use `--device cuda:0 --batch-size 32` for local models and `--concurrency 64 --requests-per-second 60` for Jev. Repeat each model on `datasets/reranker/scifact-test-top100.json.gz`. Model IDs are listed in `reranker_models.json`. `--limit 2` is available for a smoke run, but partial results cannot be published.

After all runs, export complete results and paired intervals using the same locked script environment:

```sh
uv run --no-project --script summarize_rerankers.py --records reranker-runs
```

Aggregates are `{model}--{dataset}.json`; raw per-query scores, original-index permutations, timestamps, latency, usage, and metrics are compressed under `results/leaderboard/reranker/records/`. Candidate text and labels live once in the matching hashed fixture.

To rebuild fixtures from public sources, download [LoCoMo](https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json) and [BEIR SciFact](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip), verify their hashes against the published fixture, then run:

```sh
uv run --no-project --script run_all_reranker.py prepare \
  --dataset locomo --source /path/to/locomo10.json --output /path/to/new-locomo.json.gz
uv run --no-project --script run_all_reranker.py prepare \
  --dataset scifact --source /path/to/scifact.zip --output /path/to/new-scifact.json.gz
```

Preparation refuses to overwrite a fixture. Numerical retrieval may differ across hardware; reuse the published fixture for an exact candidate comparison. Model-specific runtime metadata is preserved in each result. Hosted model IDs other than pinned Jev identify a service version but do not expose an immutable server checkpoint.

Validation: `python -m pytest tests/test_rerank_eval.py -q`, `ruff check rerank_eval.py summarize_rerankers.py run_all_reranker.py tests/test_rerank_eval.py`, and `npm ci && npm run build` in `visualizer`.
