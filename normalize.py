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

import pandas as pd

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

# ---------------------------------------------------------------------------
# Full-dataset materialization CLI
# ---------------------------------------------------------------------------
# The functions above are intentionally kept as lightweight utilities because
# baseline_pipeline.py imports them.  This CLI adds a scalable preprocessing
# stage that materializes normalized records once, so later blocking/feature
# extraction does not repeat the same regex work millions of times.

DEFAULT_CHUNK_SIZE = 250_000


def _normalize_chunk(chunk):
    """Add reusable normalized/signature columns to one TSV chunk."""
    chunk = chunk.copy()
    chunk.columns = [str(c).strip() for c in chunk.columns]

    chunk["name_norm"] = chunk["business_name"].map(normalize_name)
    chunk["address_norm"] = chunk["business_address"].map(normalize_address)

    name_parts = chunk["name_norm"].str.split()
    chunk["first_token"] = name_parts.map(lambda x: x[0] if x else "")
    chunk["last_token"] = name_parts.map(lambda x: x[-1] if x else "")
    chunk["first_two"] = name_parts.map(lambda x: " ".join(x[:2]) if x else "")
    chunk["sorted_name"] = name_parts.map(lambda x: " ".join(sorted(set(x))) if len(x) >= 2 else "")
    chunk["first_last"] = name_parts.map(lambda x: f"{x[0]}|{x[-1]}" if len(x) >= 2 else "")

    compact = chunk["name_norm"].str.replace(r"[^a-z0-9]+", "", regex=True)
    chunk["boundary"] = compact.map(
        lambda x: x[:4] + "|" + x[-4:] if len(x) >= 8 else ""
    )

    chunk["name_len"] = chunk["business_name"].str.len().astype("int32")
    chunk["address_len"] = chunk["business_address"].str.len().astype("int32")
    chunk["name_token_count"] = name_parts.map(len).astype("int16")
    chunk["address_token_count"] = chunk["address_norm"].str.split().map(len).astype("int16")
    chunk["name_digit_count"] = chunk["business_name"].str.count(r"\d").astype("int16")
    chunk["address_digit_count"] = chunk["business_address"].str.count(r"\d").astype("int16")
    return chunk


def normalize_file(input_path, output_path, chunk_size=DEFAULT_CHUNK_SIZE):
    """Normalize one large TSV into one Parquet file without loading it all."""
    import os
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required for full-dataset materialization. "
            "Install it with: pip install pyarrow"
        ) from exc

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = None
    total_rows = 0

    try:
        for chunk in pd.read_csv(
            input_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
            chunksize=chunk_size,
        ):
            out = _normalize_chunk(chunk)
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
            total_rows += len(out)
            print(f"  {os.path.basename(input_path)}: {total_rows:,} rows", flush=True)
    finally:
        if writer is not None:
            writer.close()

    return total_rows


def normalize_ground_truth_file(input_path, output_path):
    """Parse and validate train_ground_truth.tsv without changing entity IDs.

    Ground-truth IDs are identifiers, not natural-language fields, so they must
    NOT be lowercased, punctuation-normalized, or suffix-stripped. We preserve
    the original comma-separated match list exactly and add lightweight count
    columns useful for diagnostics/training.
    """
    import os
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    df.columns = [str(c).strip() for c in df.columns]

    expected = {"source1_entity_id", "matched_entity_ids"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"{input_path}: missing required columns {sorted(missing)}; "
            f"found {list(df.columns)!r}"
        )

    if df["source1_entity_id"].duplicated().any():
        dup = int(df["source1_entity_id"].duplicated().sum())
        raise ValueError(f"{input_path}: found {dup:,} duplicate source1_entity_id values")

    # Preserve the exact original label string. Empty string means singleton.
    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("").astype(str)

    def split_ids(x):
        return [v.strip() for v in x.split(",") if v.strip()] if x else []

    match_lists = df["matched_entity_ids"].map(split_ids)
    df["match_count"] = match_lists.map(len).astype("int16")
    df["is_singleton"] = (df["match_count"] == 0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, output_path, compression="zstd")

    n = len(df)
    n_singletons = int(df["is_singleton"].sum())
    print(
        f"  ground truth: {n:,} entities, {n_singletons:,} singletons, "
        f"{n - n_singletons:,} non-singletons",
        flush=True,
    )
    return n


def materialize_dataset(data_root="dataset", output_root="normalized", chunk_size=DEFAULT_CHUNK_SIZE):
    """Materialize normalized Parquet for all six source TSVs plus ground truth."""
    import json
    import os

    files = [
        ("train", "train_source1.tsv"),
        ("train", "train_source2.tsv"),
        ("train", "train_source3.tsv"),
        ("test", "test_source1.tsv"),
        ("test", "test_source2.tsv"),
        ("test", "test_source3.tsv"),
    ]

    manifest = {}

    # Ground truth is handled separately because its values are identifiers,
    # not names/addresses.
    gt_src = os.path.join(data_root, "train", "train_ground_truth.tsv")
    gt_dst = os.path.join(output_root, "train", "train_ground_truth.parquet")
    print("\n[train/train_ground_truth.tsv]", flush=True)
    if not os.path.exists(gt_src):
        raise FileNotFoundError(f"Missing ground-truth file: {gt_src}")
    n_gt = normalize_ground_truth_file(gt_src, gt_dst)
    manifest["train/train_ground_truth.tsv"] = {
        "rows": n_gt,
        "output": gt_dst,
        "type": "labels/ids-preserved",
    }

    for split, filename in files:
        src = os.path.join(data_root, split, filename)
        dst = os.path.join(output_root, split, filename.replace(".tsv", ".parquet"))
        print(f"\n[{split}/{filename}]", flush=True)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing source file: {src}")
        n = normalize_file(src, dst, chunk_size=chunk_size)
        manifest[f"{split}/{filename}"] = {
            "rows": n,
            "output": dst,
        }

    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nNormalization complete.")
    print(f"Manifest: {os.path.join(output_root, 'manifest.json')}")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Materialize normalized Parquet for the full entity-resolution dataset."
    )
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--output-root", default="normalized")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()

    materialize_dataset(
        data_root=args.data_root,
        output_root=args.output_root,
        chunk_size=args.chunk_size,
    )
