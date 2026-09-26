"""Reverse BM25 candidate retrieval into the deduplicated Source 1 table."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path

from v2_text import Record

LOG = logging.getLogger("v2.retrieval")
INDEX_VERSION = "unicode-bm25-v2"
FIELDS = ("name", "name_core", "address", "address_core", "nw", "ncw", "aw", "acw")


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)


def country_dir(root: Path, country: str) -> Path:
    return root / hashlib.sha256(country.encode()).hexdigest()[:16]


def tokenizer(index):
    import tantivy
    index.register_tokenizer("ws", tantivy.TextAnalyzerBuilder(tantivy.Tokenizer.whitespace()).build())


def build_index(source1_path: Path, root: Path, threads: int = 4) -> dict:
    """Build one country index. Identical source bytes/schema are required for cache reuse."""
    import tantivy
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source1_path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    text_digest = hashlib.sha256(Path(__file__).with_name("v2_text.py").read_bytes()).hexdigest()
    signature = {
        "version": INDEX_VERSION,
        "source_sha256": digest.hexdigest(),
        "text_code_sha256": text_digest,
        "tantivy_version": tantivy.__version__,
    }
    manifest = root / "manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if all(saved.get(key) == value for key, value in signature.items()):
            LOG.info("Reusing compatible index: %s", saved)
            return saved
        raise RuntimeError("Index cache is incompatible; use a new cache prefix")
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_unsigned_field("rid", fast=True, stored=True)
    for field in FIELDS:
        schema_builder.add_text_field(field, tokenizer_name="ws", index_option="freq")
    schema = schema_builder.build()
    writers, indexes, counts = {}, {}, {}
    for rid, row in enumerate(rows(source1_path)):
        record = Record.from_row(row)
        country = record.country
        if country not in writers:
            path = country_dir(root, country)
            path.mkdir(parents=True, exist_ok=True)
            indexes[country] = tantivy.Index(schema, path=str(path))
            tokenizer(indexes[country])
            writers[country] = indexes[country].writer(heap_size=256_000_000, num_threads=threads)
            counts[country] = 0
        document = {key: " ".join(value) for key, value in record.fields().items()}
        document["rid"] = rid
        writers[country].add_document(tantivy.Document.from_dict(document, schema))
        counts[country] += 1
        if (rid + 1) % 100000 == 0:
            LOG.info("Indexed %s reference records", rid + 1)
    for country, writer in writers.items():
        LOG.info("Committing %s index (%s records)", country, counts[country])
        writer.commit()
        writer.wait_merging_threads()
    signature["countries"] = counts
    signature["rows"] = sum(counts.values())
    manifest.write_text(json.dumps(signature, indent=2) + "\n")
    return signature


class Retriever:
    def __init__(self, root: Path):
        self.root = root
        self.indexes = {}
        self.searchers = {}

    def retrieve(self, record: Record, k: int = 12) -> dict[int, dict]:
        """Return bounded same-country candidates, or none for a new country."""
        import tantivy
        country = record.country
        if country not in self.indexes:
            path = country_dir(self.root, country)
            if not path.exists():
                return {}
            self.indexes[country] = tantivy.Index.open(str(path))
            tokenizer(self.indexes[country])
            self.searchers[country] = self.indexes[country].searcher()
        index, searcher = self.indexes[country], self.searchers[country]
        fields = record.fields()
        variants = {
            "joint": {"name": 1.0, "name_core": 1.5, "nw": 1.5, "ncw": 2.0,
                      "address": 0.8, "address_core": 1.0, "aw": 0.8, "acw": 1.0},
            "name": {"name": 1.0, "name_core": 1.2, "nw": 1.0, "ncw": 1.2},
            "address": {"address": 1.0, "address_core": 1.2, "aw": 1.0, "acw": 1.2},
        }
        result = {}
        for variant, weights in variants.items():
            clauses = [
                (tantivy.Occur.Should, tantivy.Query.boost_query(
                    tantivy.Query.term_query(index.schema, field, term, "freq"), weight
                ))
                for field, weight in weights.items()
                for term in fields[field]
            ]
            if not clauses:
                continue
            query = tantivy.Query.boolean_query(clauses)
            # Counting all matches disables the useful top-k pruning optimization.
            hits = searcher.search(query, limit=k, count=False).hits
            addresses = [address for _, address in hits]
            rids = searcher.fast_field_values("rid", addresses)
            for rank, ((score, _), rid) in enumerate(zip(hits, rids), start=1):
                entry = result.setdefault(int(rid), {})
                entry[variant + "_rank"] = rank
                entry[variant + "_score"] = float(score)
        return result
