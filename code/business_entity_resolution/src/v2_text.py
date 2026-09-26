"""Unicode-safe text views for ranked entity retrieval. No learned state."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

LEGAL = frozenset("inc incorporated corp corporation llc ltd limited pvt private plc llp sarl sas gmbh ag bv sa company co pte pty kg oy ab nv opc lim privatelimited pvtltd".split())
STOP = frozenset("a an and the of for at in on near to by with www com http https contact phone tel mobile website".split())
ADDRESS = dict(zip(
    "street road avenue boulevard highway drive lane suite building apartment north south east west floor department nagar extension sector plot number".split(),
    "st rd ave blvd hwy dr ln ste bldg apt n s e w fl dept ngr ext sec plt no".split(),
))
WEB = re.compile(r"(?:https?://|www\.)\S+|\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.I)


def normalize(value: str) -> str:
    """Fold Latin accents while retaining the letters and vowel signs of other scripts."""
    result = []
    latin = False
    for ch in unicodedata.normalize("NFKD", value.casefold()):
        category = unicodedata.category(ch)
        if category.startswith("M"):
            if not latin:
                result.append(ch)
            continue
        latin = "LATIN" in unicodedata.name(ch, "")
        result.append(ch if ch.isalnum() else " ")
    return " ".join("".join(result).split())


def romanize(value: str) -> str:
    if value.isascii():
        return value
    from anyascii import anyascii
    return normalize(anyascii(value))


def clean_name(value: str) -> str:
    words = normalize(WEB.sub(" ", value).replace("&", " and ")).split()
    return " ".join(word for word in words if word not in LEGAL and word not in STOP)


def clean_address(value: str) -> str:
    return " ".join(ADDRESS.get(word, word) for word in normalize(value).split())


def grams(value: str) -> set[str]:
    """All character trigrams; no position sampling or frequency deletion."""
    value = "_" + value.replace(" ", "_") + "_"
    return {value[i:i + 3] for i in range(len(value) - 2)} if len(value) >= 3 else set()


@dataclass(slots=True)
class Record:
    entity_id: str
    country: str
    name: str
    address: str
    roman_name: str
    roman_address: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Record":
        name = clean_name(row.get("business_name", ""))
        address = clean_address(row.get("business_address", ""))
        return cls(row["entity_id"], row["country"], name, address,
                   clean_name(romanize(name)), clean_address(romanize(address)))

    def fields(self) -> dict[str, list[str]]:
        return {
            "name": sorted(grams(self.name) | grams(self.roman_name)),
            "address": sorted(grams(self.address) | grams(self.roman_address)),
            "nw": sorted(set(self.name.split()) | set(self.roman_name.split())),
            "aw": sorted(set(self.address.split()) | set(self.roman_address.split())),
        }


def entity_partition(entity_id: str, modulus: int) -> int:
    """IDs select evaluation folds only; they never enter retrieval or model features."""
    import hashlib
    return int.from_bytes(hashlib.blake2b(entity_id.encode(), digest_size=8).digest(), "little") % modulus
