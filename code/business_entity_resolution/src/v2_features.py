"""Fast, deterministic pair features for candidate pairs only."""

from __future__ import annotations

import re
from functools import lru_cache

from rapidfuzz import fuzz

from v2_text import Record, grams

FEATURE_NAMES = (
    "name_ratio", "name_token_sort", "name_token_set", "name_partial",
    "name_core_ratio", "name_core_token_set", "name_translit_ratio",
    "name_translit_token_set", "name_translit_gram_jaccard",
    "name_gram_jaccard", "name_gram_containment", "name_token_jaccard",
    "name_token_containment", "name_common_tokens", "name_prefix_equal",
    "name_length_ratio", "name_exact", "core_name_exact",
    "address_ratio", "address_token_sort", "address_token_set",
    "address_translit_ratio", "address_translit_gram_jaccard",
    "address_gram_jaccard", "address_gram_containment", "address_token_jaccard",
    "address_token_containment", "address_common_tokens", "address_exact",
    "number_jaccard", "number_containment", "number_common", "number_conflict",
    "postal_equal", "postal_conflict", "name_missing", "address_missing",
    "target_name_length", "reference_name_length", "target_address_length",
    "reference_address_length", "name_ratio_joint", "name_ratio_address",
    "name_ratio_rank", "address_ratio_rank", "joint_rank",
    "name_rank", "address_rank", "joint_score", "name_score", "address_score",
    "best_rank", "rank_gap", "rrf_score", "target_is_s2", "country_us",
    "country_india", "country_france",
)

_NUMBERS = re.compile(r"\d+")


@lru_cache(maxsize=16000)
def _grams(value: str) -> frozenset[str]:
    return frozenset(grams(value))


@lru_cache(maxsize=16000)
def _tokens(value: str) -> frozenset[str]:
    return frozenset(value.split())


def _ratio(left: str, right: str, fn) -> float:
    if not left or not right:
        return 0.0
    return fn(left, right) / 100.0


def _postal(numbers: list[str], country: str) -> str:
    allowed = {"US": {5, 9}, "India": {6}, "France": {5}}.get(country, {5, 6})
    return next((number for number in reversed(numbers) if len(number) in allowed), "")


def _jaccard(left: set[str], right: set[str]) -> tuple[float, float]:
    if not left or not right:
        return 0.0, 0.0
    common = len(left & right)
    return common / len(left | right), common / min(len(left), len(right))


def pair_features(target: Record, reference: Record, evidence: dict, target_source: str) -> list[float]:
    tn, rn = target.raw_name or target.name, reference.raw_name or reference.name
    tc, rc = target.core_name or target.name, reference.core_name or reference.name
    tnr, rnr = target.roman_name or target.name, reference.roman_name or reference.name
    ta, ra = target.raw_address or target.address, reference.raw_address or reference.address
    tnt, rnt = _tokens(tc), _tokens(rc)
    tat, rat = _tokens(target.address), _tokens(reference.address)
    tng, rng = _grams(tc), _grams(rc)
    tag, rag = _grams(target.address), _grams(reference.address)
    name_j, name_c = _jaccard(tnt, rnt)
    addr_j, addr_c = _jaccard(tat, rat)
    ngram_j, ngram_c = _jaccard(tng, rng)
    agr_j, agr_c = _jaccard(tag, rag)
    target_number_order, reference_number_order = _NUMBERS.findall(ta), _NUMBERS.findall(ra)
    tnums, rnums = set(target_number_order), set(reference_number_order)
    common_numbers = len(tnums & rnums)
    tpost = _postal(target_number_order, target.country)
    rpost = _postal(reference_number_order, reference.country)
    jr, nr, ar = (int(evidence.get(v + "_rank", 25)) for v in ("joint", "name", "address"))
    js, ns, ads = (float(evidence.get(v + "_score", 0.0)) for v in ("joint", "name", "address"))
    max_name = max(len(tc), len(rc), 1)
    name_ratio = _ratio(tc, rc, fuzz.ratio)
    addr_ratio = _ratio(target.address, reference.address, fuzz.ratio)
    values = [
        _ratio(tn, rn, fuzz.ratio), _ratio(tc, rc, fuzz.token_sort_ratio),
        _ratio(tc, rc, fuzz.token_set_ratio), _ratio(tc, rc, fuzz.partial_ratio),
        name_ratio, _ratio(tc, rc, fuzz.token_set_ratio), _ratio(tnr, rnr, fuzz.ratio),
        _ratio(tnr, rnr, fuzz.token_set_ratio),
        _jaccard(_grams(tnr), _grams(rnr))[0],
        ngram_j, ngram_c, name_j, name_c, float(len(tnt & rnt)),
        float(bool(tc and rc and tc.split()[0] == rc.split()[0])),
        min(len(tc), len(rc)) / max_name, float(bool(tn and rn and tn == rn)),
        float(bool(tc and rc and tc == rc)),
        addr_ratio, _ratio(target.address, reference.address, fuzz.token_sort_ratio),
        _ratio(target.address, reference.address, fuzz.token_set_ratio),
        _ratio(target.roman_address, reference.roman_address, fuzz.ratio),
        _jaccard(_grams(target.roman_address), _grams(reference.roman_address))[0],
        agr_j, agr_c, addr_j, addr_c, float(len(tat & rat)),
        float(bool(ta and ra and ta == ra)),
        common_numbers / len(tnums | rnums) if tnums | rnums else 0.0,
        common_numbers / min(len(tnums), len(rnums)) if tnums and rnums else 0.0,
        float(common_numbers), float(bool(tnums and rnums and not common_numbers)),
        float(bool(tpost and rpost and tpost == rpost)),
        float(bool(tpost and rpost and tpost != rpost)),
        float(not bool(tc)), float(not bool(target.address)),
        float(len(tc)), float(len(rc)), float(len(target.address)), float(len(reference.address)),
        name_ratio * js, name_ratio * ads,
        name_ratio / jr, addr_ratio / ar,
        float(jr), float(nr), float(ar), js, ns, ads,
        float(min(jr, nr, ar)), float(sorted((jr, nr, ar))[1] - min(jr, nr, ar)),
        1.0 / (60 + jr) + 1.0 / (60 + nr) + 1.0 / (60 + ar),
        float(target_source == "S2"),
        float(target.country == "US"), float(target.country == "India"), float(target.country == "France"),
    ]
    if len(values) != len(FEATURE_NAMES):
        raise AssertionError(f"Feature schema has {len(values)} values for {len(FEATURE_NAMES)} names")
    return values
