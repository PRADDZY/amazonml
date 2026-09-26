# Improving the 0.699 submission

The submitted baseline scored 0.699 publicly and 0.70555 on its held-out split.
Its candidate recall was only 0.58191. Blocking missed 160,339 of the 175,408
false-negative links: 91.4% of missed links occurred before matching.

The implementation discarded tokens with more than 12 target postings and
trigrams with more than 16 postings, sampled only four name trigrams and three
address trigrams, and discarded all non-ASCII letters. These restrictions make
noisy names and Indian scripts especially vulnerable.

## Replacement and evidence

1. Preserve Unicode and add deterministic transliteration. Keep every trigram.
2. Index deduplicated Source 1 by country; query targets against this smaller
   reference using BM25 name, address, and combined views. Export the inverse
   mapping as Source 1 candidate lists. No Cartesian product is materialized.
3. Measure candidate recall at several k values against the complete reference
   corpus before training. A sampled-positive recall audit is explicitly not an
   end-to-end macro F0.5 estimate.
4. Fit a richer pair classifier using hard negatives, then validate target
   exclusivity and score margins. Hold out complete Source 1 groups; do not put
   IDs or fold identifiers into model features.
5. Select the smallest candidate set that preserves measured matching quality.
   The exported candidate file must include exactly the pairs sent to the final
   matcher, before the final decision threshold.

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
data indexing, training, and inference run on AWS. Local work is source editing,
small unit tests, document inspection, and artifact packaging. No external business
identities, addresses, labels, competitor models, or predictions enter the pipeline.

The user authorized up to $100 for this improvement effort on September 26,
replacing the earlier $90 ceiling. Each worker has an independent shutdown timer,
encrypted storage deleted on termination, and a finite job timeout. Only one
worker is launched at a time. Status checks are 15 minutes apart.

The 0.990788 leaderboard score is a target, not a claimed result. No new full
submission is promoted solely because its process completed successfully.
