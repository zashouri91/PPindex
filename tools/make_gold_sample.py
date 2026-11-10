"""
Draw the gold-set sample for checking the comment tagging.

    python tools/make_gold_sample.py [--n 300] [--seed 20261110] [--force]

300 substantive comments from completed responses at the 10 in-scope sites,
stratified by open-ended question x site (largest-remainder proportional
allocation, at least one per non-empty cell), fixed seed. Writes
data/gold/gold_labels.csv with comment IDs and EMPTY label columns:

    comment_id, themes, sentiment, labeler, verified, notes

IDs and labels only — never comment text. An existing file is not overwritten
(use --force), so labels already entered are never lost.

Labelling convention: themes = any of the 29 taxonomy themes, pipe-separated
("Unclassified" if none fits); sentiment = Positive / Negative / Neutral / Mixed,
where a complaint OR an unmet-need request ("would be nice to have more ...")
is Negative.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(ROOT, "data", "gold", "gold_labels.csv")
COLUMNS = ["comment_id", "themes", "sentiment", "labeler", "verified", "notes"]


def draw(comments: pd.DataFrame, ten_sites, n=300, seed=20261110) -> pd.DataFrame:
    pool = comments[(comments["reached_end"] == 1) & comments["site"].isin(ten_sites)
                    & ~comments["is_junk"].astype(bool)]
    cells = pool.groupby(["source_question", "site"]).size()
    exact = cells / cells.sum() * n
    alloc = np.maximum(1, np.floor(exact)).astype(int)
    # largest remainder until the total is n
    rem = (exact - np.floor(exact)).sort_values(ascending=False)
    i = 0
    while alloc.sum() < n:
        k = rem.index[i % len(rem)]
        if alloc[k] < cells[k]:
            alloc[k] += 1
        i += 1
    while alloc.sum() > n:   # only if the floor-of-1 rule overshot
        k = alloc[alloc > 1].index[-1]
        alloc[k] -= 1
    rng = np.random.default_rng(seed)
    picks = []
    for key, k in alloc.items():
        ids = pool[(pool["source_question"] == key[0]) & (pool["site"] == key[1])]["comment_id"].to_numpy()
        picks += list(rng.choice(ids, size=min(k, len(ids)), replace=False))
    return pd.DataFrame({"comment_id": sorted(int(x) for x in picks)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if os.path.exists(GOLD) and not a.force:
        sys.exit(f"{GOLD} exists — not overwriting labels (use --force).")
    comments = pd.read_csv(os.path.join(ROOT, "data", "comments.csv"))
    with open(os.path.join(ROOT, "data", "meta.json")) as fh:
        ten = json.load(fh)["ten_sites"]
    out = draw(comments, ten, a.n, a.seed)
    for c in COLUMNS[1:]:
        out[c] = ""
    out["verified"] = False
    os.makedirs(os.path.dirname(GOLD), exist_ok=True)
    out[COLUMNS].to_csv(GOLD, index=False)
    print(f"wrote {len(out)} comment IDs to {GOLD}")


if __name__ == "__main__":
    main()
