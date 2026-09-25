"""
Blocking recall evaluator for Amazon ML Challenge 2026.

Purpose
-------
Evaluate candidate-blocking keys against the REAL ground truth without
materializing the huge S1 x S2/S3 candidate-pair table.

This is deliberately different from blocking_v2.py:
- It reads the ground-truth matched IDs.
- It fetches only the S2/S3 rows that are actually true matches.
- It computes blocking keys on S1 validation rows and true matched rows.
- It reports how many true matches each blocking key would recover.

No giant Cartesian merge is created.

Run from student_resource/:
    python blocking_recall_eval.py --data-root dataset

Expected output includes:
    - mean true-match recall per validation entity
    - global true-match recall
    - fraction of non-singleton entities with at least one recovered match
    - union recall for combinations of keys
"""

import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name
from scoring import deterministic_split


CHUNK_SIZE = 250_000
MIN_TOKEN_LEN = 3


def load_source(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def load_ground_truth(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    gt = {}
    for row in df.itertuples(index=False):
        gt[row.source1_entity_id] = [
            x for x in row.matched_entity_ids.split(",") if x
        ]
    return gt


def make_name_features(series):
    """
    Compute several country-agnostic name blocking signatures.
    normalize_name() itself remains the canonical implementation.
    """
    values = series.astype(str).tolist()

    name_norm = []
    sorted_name = []
    first_token = []
    last_token = []
    first_last = []
    boundary = []

    for raw in values:
        n = normalize_name(raw)
        name_norm.append(n)

        toks = [t for t in n.split() if len(t) >= MIN_TOKEN_LEN]

        if toks:
            first_token.append(toks[0])
            last_token.append(toks[-1])
        else:
            first_token.append("")
            last_token.append("")

        if len(toks) >= 2:
            first_last.append(toks[0] + "|" + toks[-1])
        else:
            first_last.append("")

        if len(toks) >= 2:
            sorted_name.append(" ".join(sorted(set(toks))))
        else:
            sorted_name.append("")

        compact = re.sub(r"[^a-z0-9]+", "", n)
        if len(compact) >= 8:
            boundary.append(compact[:4] + "|" + compact[-4:])
        else:
            boundary.append("")

    return {
        "name_norm": name_norm,
        "sorted_name": sorted_name,
        "first_token": first_token,
        "last_token": last_token,
        "first_last": first_last,
        "boundary": boundary,
    }


def fetch_true_matches(source_path, needed_ids):
    """
    Read the large source file in chunks and retain only records whose
    entity_id is present in the ground-truth match lists.
    """
    if not needed_ids:
        return pd.DataFrame(
            columns=["entity_id", "business_name", "country"]
        )

    print(
        f"  scanning {os.path.basename(source_path)} in chunks "
        f"for {len(needed_ids):,} true-match IDs...",
        flush=True,
    )

    pieces = []

    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):
        hit = chunk[chunk["entity_id"].isin(needed_ids)]
        if not hit.empty:
            pieces.append(
                hit[["entity_id", "business_name", "country"]].copy()
            )

    if not pieces:
        return pd.DataFrame(
            columns=["entity_id", "business_name", "country"]
        )

    out = pd.concat(pieces, ignore_index=True)

    # Defensive check: every fetched ID should be unique.
    if out["entity_id"].duplicated().any():
        dup = int(out["entity_id"].duplicated().sum())
        raise RuntimeError(
            f"Duplicate entity IDs found while fetching true matches: {dup}"
        )

    print(f"  fetched {len(out):,} matched rows", flush=True)
    return out


def prepare_match_info(df):
    feats = make_name_features(df["business_name"])
    out = df[["entity_id", "country"]].copy()

    for k, v in feats.items():
        out[k] = v

    return out


def evaluate_key(s1_info, match_rows, gt, key_name):
    """
    Evaluate one blocking key.

    For every actual GT match, test whether the S1 entity and matched S2/S3
    record have the same key. Then aggregate:
      - pair/global recall
      - mean per-entity recall
      - entity coverage (at least one true match recoverable)
    """
    s1_by_id = s1_info.set_index("entity_id")

    per_entity_recalled = {}
    total_true = 0
    total_recovered = 0

    for row in match_rows.itertuples(index=False):
        eid = row.source1_entity_id
        matched_id = row.matched_entity_id

        s1 = s1_by_id.loc[eid]

        s1_key = str(s1[key_name]) + "||" + str(s1["country"])
        m_key = str(getattr(row, key_name)) + "||" + str(row.country)

        total_true += 1
        hit = s1_key != "" and s1_key == m_key

        if hit:
            total_recovered += 1

        rec = per_entity_recalled.get(eid)
        if rec is None:
            per_entity_recalled[eid] = [0, 0]

        per_entity_recalled[eid][1] += 1
        if hit:
            per_entity_recalled[eid][0] += 1

    recalls = []
    entity_with_any = 0

    for eid, (covered, total) in per_entity_recalled.items():
        recalls.append(covered / total if total else 0.0)
        if covered > 0:
            entity_with_any += 1

    return {
        "mean_entity_recall": (
            sum(recalls) / len(recalls) if recalls else 0.0
        ),
        "global_pair_recall": (
            total_recovered / total_true if total_true else 0.0
        ),
        "entities_with_any_true_match_recovered": entity_with_any,
        "n_non_singleton_entities": len(per_entity_recalled),
        "entity_coverage": (
            entity_with_any / len(per_entity_recalled)
            if per_entity_recalled
            else 0.0
        ),
        "total_true_pairs": total_true,
        "recovered_true_pairs": total_recovered,
    }


def evaluate_union(s1_info, match_rows, key_names):
    s1_by_id = s1_info.set_index("entity_id")

    per_entity = {}
    total_true = 0
    total_recovered = 0

    for row in match_rows.itertuples(index=False):
        eid = row.source1_entity_id
        s1 = s1_by_id.loc[eid]

        recovered = False

        for key_name in key_names:
            s1_key = str(s1[key_name]) + "||" + str(s1["country"])
            m_key = str(row[key_name]) + "||" + str(row["country"])

            if s1_key and s1_key == m_key:
                recovered = True
                break

        total_true += 1

        if eid not in per_entity:
            per_entity[eid] = [0, 0]

        per_entity[eid][1] += 1

        if recovered:
            total_recovered += 1
            per_entity[eid][0] += 1

    recalls = []
    covered_entities = 0

    for covered, total in per_entity.values():
        recalls.append(covered / total if total else 0.0)
        if covered:
            covered_entities += 1

    return {
        "mean_entity_recall": (
            sum(recalls) / len(recalls) if recalls else 0.0
        ),
        "global_pair_recall": (
            total_recovered / total_true if total_true else 0.0
        ),
        "entity_coverage": (
            covered_entities / len(per_entity)
            if per_entity else 0.0
        ),
        "covered_entities": covered_entities,
        "n_non_singleton_entities": len(per_entity),
        "total_true_pairs": total_true,
        "recovered_true_pairs": total_recovered,
    }


def build_match_rows(gt, s1_val_ids, source_prefix):
    """
    Explode only validation ground-truth matches for one source.
    """
    rows = []

    for eid in s1_val_ids:
        for mid in gt[eid]:
            if mid.startswith(source_prefix + "-"):
                rows.append(
                    {
                        "source1_entity_id": eid,
                        "matched_entity_id": mid,
                    }
                )

    return pd.DataFrame(rows)


def run(data_root, val_frac, seed):
    train_dir = os.path.join(data_root, "train")

    print("Loading S1 + ground truth...", flush=True)
    s1 = load_source(os.path.join(train_dir, "train_source1.tsv"))
    gt = load_ground_truth(
        os.path.join(train_dir, "train_ground_truth.tsv")
    )

    _, val_ids = deterministic_split(
        s1["entity_id"], val_frac=val_frac, seed=seed
    )

    s1_val = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)

    print(f"validation S1 entities: {len(s1_val):,}", flush=True)
    print("Preparing S1 name signatures...", flush=True)

    s1_features = make_name_features(s1_val["business_name"])

    s1_info = s1_val[["entity_id", "country"]].copy()
    for k, v in s1_features.items():
        s1_info[k] = v

    s2_match_pairs = build_match_rows(gt, val_ids, "S2")
    s3_match_pairs = build_match_rows(gt, val_ids, "S3")

    print(
        f"true S2 validation matches: {len(s2_match_pairs):,}",
        flush=True,
    )
    print(
        f"true S3 validation matches: {len(s3_match_pairs):,}",
        flush=True,
    )

    needed_s2 = set(s2_match_pairs["matched_entity_id"])
    needed_s3 = set(s3_match_pairs["matched_entity_id"])

    s2_true = fetch_true_matches(
        os.path.join(train_dir, "train_source2.tsv"),
        needed_s2,
    )
    s3_true = fetch_true_matches(
        os.path.join(train_dir, "train_source3.tsv"),
        needed_s3,
    )

    def attach_pairs(pair_df, true_df):
        if pair_df.empty:
            return pd.DataFrame(
                columns=[
                    "source1_entity_id",
                    "matched_entity_id",
                    "country",
                    "name_norm",
                    "sorted_name",
                    "first_token",
                    "last_token",
                    "first_last",
                    "boundary",
                ]
            )

        m = pair_df.merge(
            true_df,
            left_on="matched_entity_id",
            right_on="entity_id",
            how="left",
            suffixes=("", "_matched"),
            validate="one_to_one",
        )

        m = m.rename(columns={"country": "country"})

        # true_df feature columns are already named correctly.
        return m[
            [
                "source1_entity_id",
                "matched_entity_id",
                "country",
                "name_norm",
                "sorted_name",
                "first_token",
                "last_token",
                "first_last",
                "boundary",
            ]
        ]

    s2_eval = attach_pairs(
        s2_match_pairs, prepare_match_info(s2_true)
    )
    s3_eval = attach_pairs(
        s3_match_pairs, prepare_match_info(s3_true)
    )

    all_eval = pd.concat([s2_eval, s3_eval], ignore_index=True)

    keys = [
        "name_norm",
        "sorted_name",
        "first_token",
        "last_token",
        "first_last",
        "boundary",
    ]

    print("\n" + "=" * 78)
    print("INDIVIDUAL BLOCKING-KEY RECALL")
    print("=" * 78)

    for key in keys:
        r = evaluate_key(s1_info, all_eval, gt, key)

        print(f"\n[{key}]")
        print(f"  mean entity recall : {r['mean_entity_recall']:.4f}")
        print(f"  global pair recall : {r['global_pair_recall']:.4f}")
        print(
            f"  entity coverage    : {r['entity_coverage']:.4f}"
            f"  ({r['entities_with_any_true_match_recovered']:,}/"
            f"{r['n_non_singleton_entities']:,})"
        )

    unions = [
        ("exact + sorted", ["name_norm", "sorted_name"]),
        (
            "exact + sorted + first_last",
            ["name_norm", "sorted_name", "first_last"],
        ),
        (
            "exact + sorted + first_last + boundary",
            ["name_norm", "sorted_name", "first_last", "boundary"],
        ),
        (
            "all name keys",
            keys,
        ),
    ]

    print("\n" + "=" * 78)
    print("UNION BLOCKING-KEY RECALL")
    print("=" * 78)

    for label, key_names in unions:
        r = evaluate_union(s1_info, all_eval, key_names)

        print(f"\n[{label}]")
        print(f"  mean entity recall : {r['mean_entity_recall']:.4f}")
        print(f"  global pair recall : {r['global_pair_recall']:.4f}")
        print(
            f"  entity coverage    : {r['entity_coverage']:.4f}"
            f"  ({r['covered_entities']:,}/"
            f"{r['n_non_singleton_entities']:,})"
        )

    print("\nNOTE:")
    print(
        "These numbers measure whether a TRUE match shares a blocking key "
        "with its S1 entity. They do not yet include generic-block caps or "
        "candidate-pair counts. We will use the strongest recall/efficiency "
        "combination in the next matcher."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(args.data_root, args.val_frac, args.seed)
