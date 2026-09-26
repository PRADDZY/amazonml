"""Pure-Python entity-resolution primitives shared by Spark and local tests."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
    "pvt", "private", "plc", "llp", "sarl", "sas", "gmbh", "ag", "bv",
    "sa", "company", "co", "pte", "pty", "kg", "oy", "ab", "nv",
}
ADDRESS_ABBREVIATIONS = {
    "street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd",
    "highway": "hwy", "drive": "dr", "lane": "ln", "suite": "ste",
    "building": "bldg", "apartment": "apt", "north": "n", "south": "s",
    "east": "e", "west": "w", "floor": "fl", "department": "dept",
}
STOPWORDS = {
    "a", "an", "and", "at", "by", "for", "in", "near", "of", "on", "the",
    "to", "unit", "with",
}
FEATURE_NAMES = (
    "name_edit_similarity", "name_token_jaccard", "name_token_containment",
    "address_edit_similarity", "address_token_jaccard", "address_token_containment",
    "number_jaccard", "postal_equal", "exact_name", "exact_address",
    "postal_house_hit", "name_token_hits", "address_token_hits", "name_gram_hits",
    "address_gram_hits", "rarity_score", "retrieval_score", "target_is_s2",
    "query_address_missing", "target_address_missing", "name_length_ratio",
    "address_length_ratio",
)
PAIR_FEATURE_COLUMNS = (
    "source1_id", "target_id", "target_source", "candidate_rank", *FEATURE_NAMES,
)


def spark_s3_uri(uri: str, scheme: str = "s3a") -> str:
    """Select the S3 filesystem scheme required by the Spark runtime."""
    if scheme not in {"s3", "s3a"}:
        raise ValueError("Spark S3 scheme must be 's3' or 's3a'")
    if uri.startswith(("s3://", "s3a://")):
        return scheme + "://" + uri.split("://", 1)[1]
    return uri


def normalize_text(value: str | None) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = value.encode("ascii", "ignore").decode("ascii").casefold()
    return " ".join(re.findall(r"[a-z0-9]+", value))


def normalize_name(value: str | None) -> str:
    tokens = normalize_text(value).split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def normalize_address(value: str | None) -> str:
    tokens = normalize_text(value).split()
    return " ".join(ADDRESS_ABBREVIATIONS.get(token, token) for token in tokens)


def _trigrams(value: str, limit: int) -> set[str]:
    compact = value.replace(" ", "")
    if len(compact) < 3:
        return {compact} if compact else set()
    all_grams = [compact[index:index + 3] for index in range(len(compact) - 2)]
    if len(all_grams) <= limit:
        return set(all_grams)
    if limit == 1:
        return {all_grams[0]}
    indexes = {round(i * (len(all_grams) - 1) / (limit - 1)) for i in range(limit)}
    return {all_grams[index] for index in indexes}


def blocking_keys(name: str | None, address: str | None) -> set[tuple[str, str]]:
    normalized_name = normalize_name(name)
    normalized_address = normalize_address(address)
    keys: set[tuple[str, str]] = set()

    if normalized_name:
        keys.add(("exact_name", normalized_name))
        name_tokens = sorted(
            {token for token in normalized_name.split()
             if token not in STOPWORDS and token not in LEGAL_SUFFIXES},
            key=lambda token: (-len(token), token),
        )[:4]
        keys.update(("name_token", token) for token in name_tokens)
        keys.update(("name_gram", gram) for gram in _trigrams(normalized_name, 4))
    if normalized_address:
        keys.add(("exact_address", normalized_address))
        address_tokens = sorted(
            {token for token in normalized_address.split()
             if token not in STOPWORDS and not token.isdigit()},
            key=lambda token: (-len(token), token),
        )[:4]
        keys.update(("address_token", token) for token in address_tokens)
        keys.update(("address_gram", gram) for gram in _trigrams(normalized_address, 3))

    address_tokens = normalized_address.split()
    numeric_tokens = [token for token in address_tokens if token.isdigit()]
    postal = next((token for token in reversed(numeric_tokens) if len(token) in (5, 6)), "")
    house_number = numeric_tokens[0] if numeric_tokens else ""
    if postal and house_number:
        keys.add(("postal_house", f"{postal}|{house_number}"))
    return keys


def select_top_candidates(
    rows: Iterable[Mapping[str, Any]], cap_per_source: int,
) -> list[dict[str, Any]]:
    if cap_per_source < 0:
        raise ValueError("cap_per_source must be non-negative")
    best_by_source_id: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        item = dict(row)
        source = str(item["target_source"])
        target_id = str(item["target_id"])
        key = (source, target_id)
        old = best_by_source_id.get(key)
        if old is None or (float(item["rank"]), target_id) > (float(old["rank"]), target_id):
            best_by_source_id[key] = item

    by_source: dict[str, list[dict[str, Any]]] = {"S2": [], "S3": []}
    for (source, _), row in best_by_source_id.items():
        by_source.setdefault(source, []).append(row)
    result: list[dict[str, Any]] = []
    for source in ("S2", "S3", *sorted(set(by_source) - {"S2", "S3"})):
        ordered = sorted(by_source.get(source, []),
                         key=lambda item: (-float(item["rank"]), str(item["target_id"])))
        result.extend(ordered[:cap_per_source])
    return result


def macro_f0_5(
    truth_by_query: Sequence[set[str] | frozenset[str]],
    predicted_by_query: Sequence[set[str] | frozenset[str]],
) -> float:
    if len(truth_by_query) != len(predicted_by_query):
        raise ValueError("truth and prediction query counts must match")
    if not truth_by_query:
        return 0.0
    scores: list[float] = []
    for truth, predicted in zip(truth_by_query, predicted_by_query):
        if not truth and not predicted:
            scores.append(1.0)
            continue
        if not predicted:
            scores.append(0.0)
            continue
        true_positive = len(truth & predicted)
        precision = true_positive / len(predicted)
        recall = true_positive / len(truth) if truth else 0.0
        denominator = 0.25 * precision + recall
        scores.append(1.25 * precision * recall / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def select_operating_point(
    metrics: Sequence[Mapping[str, Any]], recall_gate: float = 0.99,
    score_tolerance: float = 0.002,
) -> dict[str, Any]:
    if not metrics:
        raise ValueError("at least one cap metric is required")
    best_score = max(float(item["macro_f0_5"]) for item in metrics)
    eligible = [item for item in metrics
                if float(item["candidate_recall"]) >= recall_gate
                and float(item["macro_f0_5"]) >= best_score - score_tolerance]
    if eligible:
        chosen = min(eligible, key=lambda item: int(item["cap_per_source"]))
    else:
        chosen = min((item for item in metrics
                      if float(item["macro_f0_5"]) == best_score),
                     key=lambda item: int(item["cap_per_source"]))
    return dict(chosen)
