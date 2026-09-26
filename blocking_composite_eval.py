"""
Composite blocking evaluator for Amazon ML Challenge 2026.

One-pass experiment:
  * uses the same deterministic 10% validation S1 sample
  * scans each full S2/S3 source once
  * evaluates true-match recall for composite name+address keys
  * measures sampled-S1 candidate volume for the same keys

No full candidate-pair table is materialized.
"""

import argparse
import hashlib
import os
import re
import time
from collections import defaultdict

import pandas as pd

from normalize import normalize_name, normalize_address, address_signal_tokens
from scoring import deterministic_split


MIN_TOKEN_LEN = 3

KEYS = [
    "name_norm__addr_digits",
    "sorted_name__addr_digits",
    "first_token__addr_digits",
    "first_last__addr_digits",
    "last_token__addr_digits",
    "name_norm__addr_first3",
    "sorted_name__addr_first3",
    "first_token__addr_first3",
    "first_last__addr_first3",
    "name_norm__addr_norm",
]


def load_source(path, chunksize=None):
    kwargs = dict(
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    if chunksize:
        kwargs["chunksize"] = chunksize
    return pd.read_csv(path, **kwargs)


def load_ground_truth(path):
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    df.columns = [str(c).strip() for c in df.columns]
    return {
        s1_id: [x for x in matched.split(",") if x]
        for s1_id, matched in zip(
            df["source1_entity_id"],
            df["matched_entity_ids"],
        )
    }


def make_features(df):
    names = df["business_name"].astype(str).map(normalize_name)
    addrs = df["business_address"].astype(str).map(normalize_address)

    first_token = []
    last_token = []
    first_last = []
    sorted_name = []

    for n in names.tolist():
        toks = [t for t in n.split() if len(t) >= MIN_TOKEN_LEN]

        if toks:
            first_token.append(toks[0])
            last_token.append(toks[-1])
        else:
            first_token.append("")
            last_token.append("")

        if len(toks) >= 2:
            first_last.append(toks[0] + "|" + toks[-1])
            sorted_name.append(" ".join(sorted(set(toks))))
        else:
            first_last.append("")
            sorted_name.append("")

    address_digits = []
    address_first3 = []

    for a in addrs.tolist():
        signal = sorted(set(address_signal_tokens(a)))
        nums = re.findall(r"\d+", a)

        address_digits.append("|".join(nums) if nums else "")
        address_first3.append(" ".join(signal[:3]) if signal else "")

    out = pd.DataFrame({
        "entity_id": df["entity_id"].astype(str).tolist(),
        "country": df["country"].astype(str).tolist(),
        "name_norm": names.tolist(),
        "sorted_name": sorted_name,
        "first_token": first_token,
        "last_token": last_token,
        "first_last": first_last,
        "address_norm": addrs.tolist(),
        "address_digits": address_digits,
        "address_first3": address_first3,
    })

    address_aliases = {
        "addr_norm": "address_norm",
        "addr_digits": "address_digits",
        "addr_first3": "address_first3",
    }

    for key in KEYS:
        name_key, addr_key = key.split("__")
        addr_col = address_aliases.get(addr_key, addr_key)

        if name_key not in out.columns:
            raise RuntimeError(
                f"Unknown name feature '{name_key}' for composite key '{key}'"
            )
        if addr_col not in out.columns:
            raise RuntimeError(
                f"Unknown address feature '{addr_col}' for composite key '{key}'"
            )

        out[key] = [
            f"{n}||{a}" if n and a else ""
            for n, a in zip(out[name_key], out[addr_col])
        ]

    return out


def sample_validation_ids(entity_ids, val_frac, sample_frac, seed):
    _, val_ids = deterministic_split(
        entity_ids,
        val_frac=val_frac,
        seed=seed,
    )

    threshold = int(sample_frac * (2**32))
    sample_seed = seed + 100003

    sample_ids = {
        eid for eid in val_ids
        if int(
            hashlib.md5(
                f"{sample_seed}-{eid}".encode()
            ).hexdigest()[:8],
            16,
        ) < threshold
    }

    if not sample_ids:
        raise RuntimeError("Sample produced zero validation entities.")

    return sample_ids


def true_match_ids(gt, sample_ids, prefix):
    return {
        mid
        for eid in sample_ids
        for mid in gt[eid]
        if mid.startswith(prefix + "-")
    }


def expected_pairs(gt, sample_ids, prefix):
    rows = []
    for eid in sample_ids:
        for mid in gt[eid]:
            if mid.startswith(prefix + "-"):
                rows.append(
                    {"source1_entity_id": eid, "matched_entity_id": mid}
                )
    return rows


def evaluate_recall(s1_info, true_rows, key):
    if not true_rows:
        return 0.0, 0.0, 0, 0

    s1_map = dict(
        zip(
            s1_info["entity_id"],
            zip(s1_info[key], s1_info["country"]),
        )
    )

    total_by_entity = defaultdict(int)
    recovered_by_entity = defaultdict(int)

    for row in true_rows:
        eid = row["source1_entity_id"]
        total_by_entity[eid] += 1

        sv, sc = s1_map.get(eid, ("", ""))
        if (
            row[key]
            and sv
            and row[key] == sv
            and row["country"] == sc
        ):
            recovered_by_entity[eid] += 1

    recalls = []
    for eid, total in total_by_entity.items():
        recalls.append(recovered_by_entity[eid] / total)

    recovered = sum(recovered_by_entity.values())
    total = sum(total_by_entity.values())
    covered = sum(v > 0 for v in recovered_by_entity.values())

    return (
        sum(recalls) / len(recalls) if recalls else 0.0,
        recovered / total if total else 0.0,
        covered,
        len(total_by_entity),
    )


def evaluate_union_recall(s1_info, true_rows, key_names):
    if not true_rows:
        return 0.0, 0.0, 0, 0

    s1 = s1_info.set_index("entity_id")

    total_by_entity = defaultdict(int)
    recovered_by_entity = defaultdict(int)

    for row in true_rows:
        eid = row["source1_entity_id"]
        total_by_entity[eid] += 1

        sr = s1.loc[eid]
        hit = False

        for key in key_names:
            if (
                sr[key]
                and row[key]
                and sr[key] == row[key]
                and sr["country"] == row["country"]
            ):
                hit = True
                break

        if hit:
            recovered_by_entity[eid] += 1

    recalls = [
        recovered_by_entity[eid] / total
        for eid, total in total_by_entity.items()
    ]
    recovered = sum(recovered_by_entity.values())
    total = sum(total_by_entity.values())
    covered = sum(v > 0 for v in recovered_by_entity.values())

    return (
        sum(recalls) / len(recalls) if recalls else 0.0,
        recovered / total if total else 0.0,
        covered,
        len(total_by_entity),
    )


def scan_source(
    path,
    sample_s1,
    needed_ids,
    candidate_key_sets,
    chunk_size,
):
    freq = {
        key: defaultdict(int)
        for key in KEYS
    }

    true_feature_rows = []

    # Only these composite values can possibly contribute candidates for
    # the sampled S1 entities.
    wanted = {}
    for key in KEYS:
        wanted[key] = {
            value + "||" + country
            for value, country in zip(
                sample_s1[key].astype(str),
                sample_s1["country"].astype(str),
            )
            if value.strip()
        }

    t0 = time.time()
    rows = 0

    print(
        f"\nScanning {os.path.basename(path)}...",
        flush=True,
    )

    for chunk_no, chunk in enumerate(
        load_source(path, chunksize=chunk_size),
        start=1,
    ):
        rows += len(chunk)

        feats = make_features(chunk)

        # Candidate-volume counts.
        for key in KEYS:
            wanted_key = wanted[key]
            if not wanted_key:
                continue

            vals = feats[key].astype(str)
            countries = feats["country"].astype(str)
            valid = vals.str.strip() != ""

            if valid.any():
                composite = vals[valid] + "||" + countries[valid]
                mask = composite.isin(wanted_key)

                if mask.any():
                    counts = composite[mask].value_counts(sort=False)
                    for comp, count in counts.items():
                        freq[key][comp] += int(count)

        # True-match rows needed for recall.
        if needed_ids:
            hit = feats[feats["entity_id"].isin(needed_ids)]
            if not hit.empty:
                true_feature_rows.append(hit)

        if chunk_no % 4 == 0:
            elapsed = max(time.time() - t0, 1e-9)
            print(
                f"  processed {rows:,} rows "
                f"({rows / elapsed:,.0f} rows/s)",
                flush=True,
            )

    if true_feature_rows:
        true_features = pd.concat(
            true_feature_rows,
            ignore_index=True,
        )
    else:
        true_features = pd.DataFrame(
            columns=["entity_id", "country"] + KEYS
        )

    print(
        f"  finished {rows:,} rows in "
        f"{time.time() - t0:.1f}s; "
        f"captured {len(true_features):,} true-match rows",
        flush=True,
    )

    return freq, true_features


def candidate_summary(sample_s1, freq2, freq3):
    records = []

    for key in KEYS:
        counts = []

        for value, country in zip(
            sample_s1[key].astype(str),
            sample_s1["country"].astype(str),
        ):
            if not value:
                counts.append(0)
                continue

            comp = value + "||" + country
            counts.append(
                freq2[key].get(comp, 0)
                + freq3[key].get(comp, 0)
            )

        ser = pd.Series(counts, dtype="int64")
        nonzero = ser[ser > 0]

        records.append({
            "key": key,
            "s1_with_candidates": int((ser > 0).sum()),
            "coverage": float((ser > 0).mean()),
            "candidate_pairs_sample": int(ser.sum()),
            "mean_candidates_per_s1": float(ser.mean()),
            "median_nonzero": float(nonzero.median())
            if len(nonzero) else 0.0,
            "p90_nonzero": float(nonzero.quantile(0.90))
            if len(nonzero) else 0.0,
            "p95_nonzero": float(nonzero.quantile(0.95))
            if len(nonzero) else 0.0,
            "max_candidates_per_s1": int(ser.max())
            if len(ser) else 0,
            "max_s2_block": max(freq2[key].values())
            if freq2[key] else 0,
            "max_s3_block": max(freq3[key].values())
            if freq3[key] else 0,
        })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--sample-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument(
        "--output-prefix",
        default="blocking_composite",
    )
    args = parser.parse_args()

    train_dir = os.path.join(args.data_root, "train")

    print("Loading S1 + ground truth...", flush=True)

    s1 = load_source(
        os.path.join(train_dir, "train_source1.tsv")
    )
    gt = load_ground_truth(
        os.path.join(train_dir, "train_ground_truth.tsv")
    )

    sample_ids = sample_validation_ids(
        s1["entity_id"],
        args.val_frac,
        args.sample_frac,
        args.seed,
    )

    s1_sample = s1[
        s1["entity_id"].isin(sample_ids)
    ].reset_index(drop=True)

    print(
        f"sampled validation fold: {len(s1_sample):,} entities "
        f"({args.sample_frac:.1%} of validation fold)",
        flush=True,
    )

    print(
        "Preparing S1 name + address + composite signatures...",
        flush=True,
    )
    s1_info = make_features(s1_sample)

    s2_expected = expected_pairs(gt, sample_ids, "S2")
    s3_expected = expected_pairs(gt, sample_ids, "S3")

    s2_ids = true_match_ids(gt, sample_ids, "S2")
    s3_ids = true_match_ids(gt, sample_ids, "S3")

    s2_freq, s2_true = scan_source(
        os.path.join(train_dir, "train_source2.tsv"),
        s1_info,
        s2_ids,
        KEYS,
        args.chunk_size,
    )

    s3_freq, s3_true = scan_source(
        os.path.join(train_dir, "train_source3.tsv"),
        s1_info,
        s3_ids,
        KEYS,
        args.chunk_size,
    )

    def convert_true_rows(expected, captured):
        if captured.empty:
            return []

        captured = captured.drop_duplicates(
            subset=["entity_id"]
        )

        lookup = captured.set_index("entity_id")

        rows = []
        for pair in expected:
            mid = pair["matched_entity_id"]
            if mid not in lookup.index:
                continue

            row = lookup.loc[mid]
            d = {
                "source1_entity_id": pair["source1_entity_id"],
                "matched_entity_id": mid,
                "country": str(row["country"]),
                "name_norm": str(row["name_norm"]),
            }
            for key in KEYS:
                d[key] = str(row[key])
            rows.append(d)

        return rows

    s2_true_rows = convert_true_rows(
        s2_expected,
        s2_true,
    )
    s3_true_rows = convert_true_rows(
        s3_expected,
        s3_true,
    )
    true_rows = s2_true_rows + s3_true_rows

    print("\n" + "=" * 110)
    print("COMPOSITE BLOCKING — TRUE MATCH RECALL")
    print("=" * 110)

    recall_records = []

    for key in KEYS:
        mean_recall, pair_recall, covered, n = evaluate_recall(
            s1_info,
            true_rows,
            key,
        )
        recall_records.append({
            "key": key,
            "mean_entity_recall": mean_recall,
            "global_pair_recall": pair_recall,
            "entity_coverage": covered / n if n else 0.0,
        })

        print(f"\n[{key}]")
        print(f"  mean entity recall : {mean_recall:.4f}")
        print(f"  global pair recall : {pair_recall:.4f}")
        print(
            f"  entity coverage    : "
            f"{covered / n:.4f} ({covered:,}/{n:,})"
        )

    unions = [
        (
            "name_norm + digit composites",
            [
                "name_norm",
                "name_norm__addr_digits",
                "sorted_name__addr_digits",
                "first_token__addr_digits",
            ],
        ),
        (
            "name_norm + all digit composites",
            [
                "name_norm",
                "name_norm__addr_digits",
                "sorted_name__addr_digits",
                "first_token__addr_digits",
                "first_last__addr_digits",
                "last_token__addr_digits",
            ],
        ),
        (
            "name_norm + digit + first3 composites",
            [
                "name_norm",
                "name_norm__addr_digits",
                "sorted_name__addr_digits",
                "first_token__addr_digits",
                "first_last__addr_digits",
                "name_norm__addr_first3",
                "sorted_name__addr_first3",
                "first_token__addr_first3",
            ],
        ),
        (
            "all composite keys",
            KEYS,
        ),
    ]

    print("\n" + "=" * 110)
    print("COMPOSITE BLOCKING — UNION RECALL")
    print("=" * 110)

    union_records = []

    for label, union_keys in unions:
        # name_norm is not a composite key, so it is not in the data created
        # by KEYS. Add it to S1/true rows on the fly.
        if "name_norm" in union_keys:
            expanded_s1 = s1_info
            expanded_true = true_rows
        else:
            expanded_s1 = s1_info
            expanded_true = true_rows

        mean_recall, pair_recall, covered, n = evaluate_union_recall(
            expanded_s1,
            expanded_true,
            union_keys,
        )
        union_records.append({
            "strategy": label,
            "mean_entity_recall": mean_recall,
            "global_pair_recall": pair_recall,
            "entity_coverage": covered / n if n else 0.0,
        })

        print(f"\n[{label}]")
        print(f"  mean entity recall : {mean_recall:.4f}")
        print(f"  global pair recall : {pair_recall:.4f}")
        print(
            f"  entity coverage    : "
            f"{covered / n:.4f} ({covered:,}/{n:,})"
        )

    # Candidate volume is measured for each composite key. name_norm's
    # volume was already measured by blocking_candidate_profile.py.
    volume = candidate_summary(
        s1_info,
        s2_freq,
        s3_freq,
    )

    print("\n" + "=" * 120)
    print("COMPOSITE BLOCKING — CANDIDATE VOLUME")
    print("=" * 120)
    print(
        volume.to_string(
            index=False,
            formatters={
                "coverage": "{:.4f}".format,
                "mean_candidates_per_s1": "{:.1f}".format,
                "median_nonzero": "{:.1f}".format,
                "p90_nonzero": "{:.1f}".format,
                "p95_nonzero": "{:.1f}".format,
            },
        )
    )

    recall_path = args.output_prefix + "_recall.csv"
    union_path = args.output_prefix + "_union_recall.csv"
    volume_path = args.output_prefix + "_volume.csv"

    pd.DataFrame(recall_records).to_csv(recall_path, index=False)
    pd.DataFrame(union_records).to_csv(union_path, index=False)
    volume.to_csv(volume_path, index=False)

    print(f"\nSaved: {recall_path}")
    print(f"Saved: {union_path}")
    print(f"Saved: {volume_path}")
    print(
        "\nThe volume figures are for the 10% validation S1 sample and "
        "are raw per-key candidates before deduplication."
    )


if __name__ == "__main__":
    main()
