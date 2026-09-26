# Improving the 0.699 submission

The submitted baseline scored 0.699 publicly and 0.70555 on its held-out split.
Its candidate recall was only 0.58191. Blocking missed 160,339 of the 175,408
false-negative links: 91.4% of missed links occurred before matching.

The implementation discarded tokens with more than 12 target postings and
trigrams with more than 16 postings, sampled only four name trigrams and three
address trigrams, and discarded all non-ASCII letters. These restrictions make
noisy names and Indian scripts especially vulnerable.

## Replacement architecture

1. Preserve native scripts, fold Latin accents, and add deterministic ASCII
   transliteration. Index raw and legal-suffix-cleaned name/address views, word
   terms, and character trigrams in separate country indexes.
2. Query each target record against the smaller, deduplicated Source 1 index
   with flat BM25 term unions for joint, name, and address evidence. Reciprocal
   rank fusion makes at most eight Source 1 candidates per target; the inverse
   candidate mapping is written per Source 1 entity. No Cartesian product is
   materialized.
3. Audit retrieval against the full Source 1 index, then build candidate-only
   pair features for names, transliterations, addresses, house numbers, postal
   codes, country/source, and retrieval ranks.
4. Fit CPU LightGBM models on AWS. Positive targets use their owning Source 1
   entity's fold; unmatched targets use a deterministic target-ID fold. Every
   candidate for one target is scored by the same held-out model. Training rows
   exclude the held-out target and reference folds, while the fifth Source 1
   reference fold is excluded from all training splits. IDs select folds only
   and never enter model features.
5. For every held-out target, select at most one reference candidate. Tune the
   candidate cap and confidence threshold against the exact per-Source-1 macro
   F0.5, including entities with no true or predicted matches. The submission
   candidate file contains the full pre-threshold candidate set seen by the
   matcher; every predicted match must be in that set. Four folds select the
   operating point; the fifth fold reports a separate entity-level audit score.

References informing the design:

- [Sparkly, PVLDB 2023](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf):
  ranked lexical retrieval and querying the larger table against the smaller one.
- [SC-Block](https://arxiv.org/abs/2303.03132): contrastive embeddings are an
  alternative if lexical retrieval misses links that require semantic matching.
- [Ditto](https://arxiv.org/abs/2004.00584): transformer pair matching is a possible
  second-stage option if feature-based error analysis supports the added cost.
- [Tantivy Python API](https://github.com/quickwit-oss/tantivy-py): bounded top-k
  search, with result counting disabled to permit efficient retrieval.

The first experiment is a CPU retrieval audit, not an LLM deployment. All real
data indexing, training, and inference run on AWS. No external business identities,
addresses, labels, competitor models, or predictions enter the pipeline. Three
worker attempts stopped before reading challenge data: the first exposed an
archive extraction path error, the second exited during Python environment
setup, and the third showed that Windows ZIP separators were not normalized on
Linux. The bootstrap now installs dependencies into an isolated package
directory, safely normalizes archive paths, and logs the exact failing step. A
fourth retrieval pilot started at 2026-09-26 18:23:51 UTC and succeeded. On a
deterministic sample of 15,120 labeled US and India targets, the union of joint,
name, and address retrieval had 99.418% recall when accepting results ranked in
the top 24 of any view. Per-country recall was 99.548% for US and 99.223% for
India. This measures retrieval only; it is not the final capped-candidate recall,
held-out macro F0.5, or leaderboard score.

The first full V2 launch was rejected because `r6i.16xlarge` required 64 standard
EC2 vCPUs while `us-east-1` had quota 8. That attempt was terminated; the later
request for a quota increase does not gate the split workflow. The split launcher
assigns target records to deterministic, disjoint feature and test shards across
regions, merges every feature shard before a single cross-fit model and global
operating-point selection, then merges test outputs by Source 1 entity. This
keeps the original training and evaluation procedure intact while using the
independent regional EC2 quotas. Shards must all report success before the next
phase starts. The 0.990788 leaderboard score remains an unverified target until
a final submission is scored.

The user authorized use of the available $100 AWS credits for this improvement
effort, replacing the earlier $90 ceiling. Each worker has an independent shutdown
timer, encrypted storage deleted on termination, and a finite job timeout. Worker
status is checked at 15-minute intervals, with local work continuing between checks.

The 0.990788 leaderboard score is a target, not a claimed result. No new full
submission is promoted solely because its process completed successfully.
