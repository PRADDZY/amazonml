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
    raw_name: str = ""
    raw_address: str = ""
    core_name: str = ""
    roman_core_name: str = ""

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Record":
        raw_name = normalize(WEB.sub(" ", row.get("business_name", "")).replace("&", " and "))
        raw_address = clean_address(WEB.sub(" ", row.get("business_address", "")))
        name = clean_name(raw_name)
        roman_name = romanize(raw_name)
        address = raw_address
        roman_address = clean_address(romanize(raw_address))
        return cls(
            row["entity_id"], row["country"], name, address,
            clean_name(roman_name), roman_address, raw_name, raw_address,
            clean_name(raw_name), clean_name(roman_name),
        )

    def fields(self) -> dict[str, list[str]]:
        roman_raw_name = romanize(self.raw_name or self.name)
        roman_raw_address = romanize(self.raw_address or self.address)
        return {
            "name": sorted(grams(self.raw_name or self.name) | grams(roman_raw_name)),
            "name_core": sorted(grams(self.core_name or self.name) | grams(self.roman_core_name or self.roman_name)),
            "address": sorted(grams(self.raw_address or self.address) | grams(roman_raw_address)),
            "address_core": sorted(grams(self.address) | grams(self.roman_address)),
            "nw": sorted(set((self.raw_name or self.name).split()) | set(roman_raw_name.split())),
            "ncw": sorted(set((self.core_name or self.name).split()) | set((self.roman_core_name or self.roman_name).split())),
            "aw": sorted(set((self.raw_address or self.address).split()) | set(roman_raw_address.split())),
            "acw": sorted(set(self.address.split()) | set(self.roman_address.split())),
        }


def entity_partition(entity_id: str, modulus: int) -> int:
    """IDs select evaluation folds only; they never enter retrieval or model features."""
    import hashlib
    return int.from_bytes(hashlib.blake2b(entity_id.encode(), digest_size=8).digest(), "little") % modulus
