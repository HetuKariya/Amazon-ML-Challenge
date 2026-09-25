"""
Normalization utilities for business names and addresses.

Kept deliberately country-agnostic: no regex that assumes a US ZIP or
Indian PIN code format, since the test set includes France (unseen in
training) and any country-specific parsing here will silently degrade
on it. Legal-suffix stripping includes a few common US/India/France
forms as a starting point -- expand this list once you've eyeballed
false negatives from the matcher.
"""

import re

# Legal-entity suffixes to strip from names before comparison. Ordered
# longest-first so e.g. "private limited" matches before "limited" eats
# only half of it.
LEGAL_SUFFIXES = [
    "incorporated", "corporation", "private limited", "limited liability company",
    "llc", "l l c", "llp", "l l p", "ltd", "pvt", "private", "plc",
    "corp", "inc", "co", "company", "pc",
    # France
    "sasu", "sarl", "eurl", "eirl", "sas", "sa",
]
_suffix_pattern = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\b\.?"
)
_punct_pattern = re.compile(r"[^\w\s]", flags=re.UNICODE)
_ws_pattern = re.compile(r"\s+")

# Common English address-word abbreviations. Applied both ways isn't
# needed -- we normalize toward the abbreviation so "Street" and "St"
# collapse to the same token.
ADDRESS_ABBREV = {
    "street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "court": "ct", "circle": "cir",
    "place": "pl", "highway": "hwy", "apartment": "apt", "building": "bldg",
    "floor": "fl", "suite": "ste",
}
_addr_abbrev_pattern = re.compile(r"\b(" + "|".join(ADDRESS_ABBREV.keys()) + r")\b")


def normalize_name(name) -> str:
    if name is None:
        return ""
    s = str(name).lower()
    s = s.replace("&", " and ")
    s = _punct_pattern.sub(" ", s)
    s = _suffix_pattern.sub(" ", s)
    s = _ws_pattern.sub(" ", s).strip()
    return s


def normalize_address(addr) -> str:
    if addr is None:
        return ""
    s = str(addr).lower()
    s = _punct_pattern.sub(" ", s)
    s = _addr_abbrev_pattern.sub(lambda m: ADDRESS_ABBREV[m.group(1)], s)
    s = _ws_pattern.sub(" ", s).strip()
    return s


def name_tokens(name) -> frozenset:
    return frozenset(normalize_name(name).split())


def address_tokens(addr) -> frozenset:
    return frozenset(normalize_address(addr).split())


ADDRESS_STOPWORDS = frozenset(ADDRESS_ABBREV.values()) | frozenset({"near", "no", "of", "the", "and"})


def address_signal_tokens(addr) -> frozenset:
    """Address tokens with generic, near-universal words removed --
    street-type abbreviations (st, ave, rd, ...) and common landmark-
    reference fillers ("Near ...") appear in nearly every address
    regardless of which business it belongs to, so leaving them in
    inflates Jaccard similarity between addresses that aren't actually
    related. Street numbers and place names are kept -- those ARE
    discriminative."""
    return frozenset(t for t in address_tokens(addr) if t not in ADDRESS_STOPWORDS)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0