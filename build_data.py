"""
Build processed datasets for the PulsePoint open-ended analyzer.

Reads the raw survey export, produces:
  data/comments.csv        - one row per open-ended comment, tagged
  data/gap_overall.csv     - importance vs support/satisfaction gap per item
  data/gap_by_site.csv     - same, broken out by site
  data/meta.json           - filter option lists + counts

Run:  python build_data.py [raw_xlsx_path]      (default: data/raw_deid.xlsx)

Identifying columns (network-ID contact question, IP, location, browser, OS,
response date, referring sites) are dropped on load whatever file is given —
see deid.py and tools/deidentify_raw.py.
"""
import sys, os, json, re
import pandas as pd
import numpy as np
from taxonomy import tag_themes, is_junk, CATEGORIES, THEMES, THEME_TO_CATEGORY
from sentiment import score_sentiment  # domain-tuned ensemble (VADER + facilities lexicon)
from deid import drop_pii_columns

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "raw_deid.xlsx")

# ---------------------------------------------------------------------------
# 1. Load (identifying columns are never carried past this point)
# ---------------------------------------------------------------------------
df = drop_pii_columns(pd.read_excel(RAW, sheet_name=0))
cols = list(df.columns)

def find(prefix):
    for c in cols:
        if str(c).strip().startswith(prefix):
            return c
    return None

# Segmentation columns
SEG = {
    "site":   find("1. Please Select your home PulsePoint"),
    "role":   find("2. Which best describes your role"),
    "hybrid": find("3. To which hybrid category"),
    "tenure": find("4. How long have you worked"),
}
RESP = cols[0]  # Respondent

# Open-ended questions -> (short label, domain)
OPEN_ENDED = [
    (find("9. 4. Are there any important types of work"), "Work Types Not Supported", "Focused / Private Work"),
    (find("13. 8. What physical aspects"),                 "Physical Limitations",     "Physical Environment"),
    (find("17. 12. Are there any workplace services"),     "Services (improve/hinder)","Amenities & Services"),
    (find("20. 15. Are there any technology"),             "Technology (improve/hinder)","Technology"),
    (find("21. 16. Is there anything else"),               "Overall Experience",       "Policy / Experience"),
]

# ---------------------------------------------------------------------------
# 2. Rating scales -> numeric (for gap analysis)
# ---------------------------------------------------------------------------
IMPORTANCE = {"Least Important":1, "Not As Important":2, "Important":3,
              "Very Important":4, "Extremely Important":5}
SUPPORT = {"Does Not Support":1, "Rarely Supports":2, "Neutral / Sometimes Supports":3,
           "Mostly Supports":4, "Fully Supports":5}
SATISFACTION = {"Very Dissatisfied":1, "Dissatisfied":2, "Neutral":3,
                "Satisfied":4, "Very Satisfied":5}

def block(prefix, n):
    """Return ordered list of columns like '6.1.', '6.2.', ... present in data."""
    out = []
    for i in range(1, n+1):
        pat = f"{prefix}.{i}."
        col = next((c for c in cols if str(c).startswith(pat)), None)
        if col:
            # clean item label: strip leading 'X.Y.' number
            label = re.sub(r"^\s*\d+\.\d+\.\s*", "", str(col)).strip()
            out.append((col, label))
    return out

# (importance_block, delivery_block, delivery_map, domain)
PAIRS = [
    (block("6", 7),  block("7", 7),  SUPPORT,      "Work Types",          "Focused / Private Work"),
    (block("10", 9), block("11", 9), SATISFACTION, "Physical Space",      "Physical Environment"),
    (block("14", 8), block("15", 8), SATISFACTION, "Workplace Services",  "Amenities & Services"),
]

def to_num(series, mapping):
    return series.map(lambda v: mapping.get(str(v).strip()) if pd.notna(v) else None)

def classify(gap):
    if gap >= 1.0:  return "Severely Under-supported"
    if gap >= 0.5:  return "Under-supported"
    if gap >= 0.15: return "Slightly Under-supported"
    if gap <= -0.15:return "Over-supported"
    return "Well-matched"

site_col = SEG["site"]

# ---- completion flag: 'Reached end' == 1 means the respondent finished --------
REACHED = find("Reached end")
df["_reached"] = pd.to_numeric(df[REACHED], errors="coerce").fillna(0).astype(int) if REACHED else 1

def compute_gaps(sub):
    """Compute overall + by-site importance/support gaps for a subset of respondents."""
    gap_rows, gap_site_rows = [], []
    for imp_block, del_block, del_map, block_name, domain in PAIRS:
        for (imp_col, imp_label), (del_col, del_label) in zip(imp_block, del_block):
            imp_num = to_num(sub[imp_col], IMPORTANCE)
            del_num = to_num(sub[del_col], del_map)
            valid = imp_num.notna() & del_num.notna()
            if int(valid.sum()) == 0:
                continue
            mi, md = float(imp_num[valid].mean()), float(del_num[valid].mean())
            gap_rows.append({"block": block_name, "domain": domain, "item": imp_label,
                             "mean_importance": round(mi, 3), "mean_delivery": round(md, 3),
                             "gap": round(mi - md, 3), "n": int(valid.sum())})
            tmp = pd.DataFrame({"site": sub[site_col], "imp": imp_num, "del": del_num}).dropna(subset=["imp", "del"])
            for site, g in tmp.groupby("site"):
                if len(g) < 5:
                    continue
                gi, gd = g["imp"].mean(), g["del"].mean()
                gap_site_rows.append({"block": block_name, "domain": domain, "item": imp_label,
                                      "site": site, "mean_importance": round(float(gi), 3),
                                      "mean_delivery": round(float(gd), 3),
                                      "gap": round(float(gi - gd), 3), "n": int(len(g))})
    gdf = pd.DataFrame(gap_rows); gdf["classification"] = gdf["gap"].map(classify)
    gsdf = pd.DataFrame(gap_site_rows); gsdf["classification"] = gsdf["gap"].map(classify)
    return gdf, gsdf

# Default population = completers (Reached end == 1); "_all" = everyone
gap_df, gap_site_df = compute_gaps(df[df["_reached"] == 1])
gap_all_df, gap_site_all_df = compute_gaps(df)
gap_df.to_csv(os.path.join(DATA, "gap_overall.csv"), index=False)
gap_site_df.to_csv(os.path.join(DATA, "gap_by_site.csv"), index=False)
gap_all_df.to_csv(os.path.join(DATA, "gap_overall_all.csv"), index=False)
gap_site_all_df.to_csv(os.path.join(DATA, "gap_by_site_all.csv"), index=False)

# ===========================================================================
# 2b. OFFICIAL Tableau-spec scores: Friction, Composite, RYG, Quadrant, weights
#     Mirrors "Tableau Survey Analytics & Dashboard Design Specification".
# ===========================================================================
AGREEMENT = {"Strongly Disagree":1, "Disagree":2, "Neutral":3, "Agree":4, "Strongly Agree":5}
TEN_SITES = ["CA-Woodland Hills", "FL-Miami", "FL-Tampa", "GA-Atlanta",
             "IL-Chicago-Willis Tower", "IN-Indianapolis", "NY-New York (Penn1)",
             "OH-Mason", "VA-Norfolk", "VA-Richmond"]
# dimension: (importance_block|None, delivery_block, delivery_map, dimension_name, n_items)
DIMENSIONS = [
    (block("6", 7),  block("7", 7),  SUPPORT,      "Work Activity",   7),
    (block("10", 9), block("11", 9), SATISFACTION, "Physical Feature", 9),
    (block("14", 8), block("15", 8), SATISFACTION, "Service Level",   8),
    (None,           block("18", 6), AGREEMENT,    "Digital & Tech",  6),
]
DIM_WEIGHTS = {"Work Activity": 35, "Service Level": 25, "Physical Feature": 20, "Digital & Tech": 20}

def ryg_class(support_minus_importance):
    v = support_minus_importance
    if v < -0.049:  return "Under Supported"
    if v <= 1.5:    return "Target"
    return "Over Supported"

def to100(numeric_series):
    return (numeric_series - 1) / 4 * 100

def official_scores(sub):
    """Per-item + per-site Friction/Composite/RYG for a respondent subset (10 sites)."""
    item_rows, site_rows, dim_pool = [], [], {}
    for imp_block, del_block, del_map, dim, n_items in DIMENSIONS:
        has_imp = imp_block is not None
        for idx, (del_col, item) in enumerate(del_block):
            imp_col = imp_block[idx][0] if has_imp else None
            sup = to_num(sub[del_col], del_map)
            sup_ok = sup.notna()
            sup100 = to100(sup)
            composite = float(sup100[sup_ok].mean()) if sup_ok.any() else np.nan
            dim_pool.setdefault(dim, []).append(sup100[sup_ok])
            if has_imp:
                imp = to_num(sub[imp_col], IMPORTANCE)
                both = imp.notna() & sup_ok
                imp100 = to100(imp)
                fric_ind = (imp100 - sup100)[both].clip(lower=0)
                friction = float(fric_ind.mean()) if both.any() else np.nan
                fpct = float((fric_ind > 0).mean() * 100) if both.any() else np.nan
                mean_imp = float(imp[both].mean()); mean_sup = float(sup[both].mean())
                ryg = ryg_class(mean_sup - mean_imp)
                n = int(both.sum())
            else:  # Digital & Tech: friction = 100 - support, no importance
                friction = float((100 - sup100)[sup_ok].mean()) if sup_ok.any() else np.nan
                fpct = float((sup100 < 100)[sup_ok].mean() * 100) if sup_ok.any() else np.nan
                mean_imp = np.nan; mean_sup = float(sup[sup_ok].mean())
                ryg = ""; n = int(sup_ok.sum())
            row = {"dimension": dim, "item": item, "n": n,
                   "mean_importance": round(mean_imp, 3) if has_imp else "",
                   "mean_support": round(mean_sup, 3),
                   "importance_100": round(to100(imp[both]).mean(), 1) if has_imp and both.any() else "",
                   "support_100": round(composite, 1) if pd.notna(composite) else "",
                   "composite": round(composite, 1) if pd.notna(composite) else "",
                   "friction": round(friction, 1) if pd.notna(friction) else "",
                   "friction_pct": round(fpct, 0) if pd.notna(fpct) else "",
                   "ryg": ryg}
            item_rows.append(row)
            # per site
            for site, g in sub.groupby(site_col):
                gsup = to_num(g[del_col], del_map); gok = gsup.notna(); gsup100 = to100(gsup)
                comp_s = float(gsup100[gok].mean()) if gok.any() else np.nan
                if has_imp:
                    gimp = to_num(g[imp_col], IMPORTANCE); gboth = gimp.notna() & gok
                    if not gboth.any():
                        continue
                    fr = float((to100(gimp) - gsup100)[gboth].clip(lower=0).mean())
                    fp = float(((to100(gimp) - gsup100)[gboth].clip(lower=0) > 0).mean() * 100)
                    ry = ryg_class(float(gsup[gboth].mean()) - float(gimp[gboth].mean()))
                    site_rows.append({"dimension": dim, "item": item, "site": site,
                        "n": int(gboth.sum()), "mean_importance": round(float(gimp[gboth].mean()), 3),
                        "mean_support": round(float(gsup[gboth].mean()), 3),
                        "importance_100": round(to100(gimp[gboth]).mean(), 1),
                        "support_100": round(comp_s, 1), "composite": round(comp_s, 1),
                        "friction": round(fr, 1), "friction_pct": round(fp, 0), "ryg": ry})
                else:
                    if not gok.any():
                        continue
                    site_rows.append({"dimension": dim, "item": item, "site": site,
                        "n": int(gok.sum()), "mean_importance": "", "mean_support": round(float(gsup[gok].mean()), 3),
                        "importance_100": "", "support_100": round(comp_s, 1), "composite": round(comp_s, 1),
                        "friction": round(float((100 - gsup100)[gok].mean()), 1),
                        "friction_pct": round(float((gsup100 < 100)[gok].mean() * 100), 0), "ryg": ""})
    # dimension composites (pooled avg support#) + weighted overall
    dim_rows = []
    for dim, series_list in dim_pool.items():
        allv = pd.concat(series_list) if series_list else pd.Series(dtype=float)
        dim_rows.append({"dimension": dim, "composite": round(float(allv.mean()), 1) if len(allv) else np.nan,
                         "weight": DIM_WEIGHTS[dim]})
    dim_df = pd.DataFrame(dim_rows)
    tot = dim_df["weight"].sum()
    weighted = float((dim_df["composite"] * dim_df["weight"] / tot).sum())
    nonweighted = float(dim_df["composite"].mean())
    dim_df = pd.concat([dim_df, pd.DataFrame([
        {"dimension": "OVERALL (weighted)", "composite": round(weighted, 1), "weight": tot},
        {"dimension": "OVERALL (equal)", "composite": round(nonweighted, 1), "weight": ""}])],
        ignore_index=True)
    return pd.DataFrame(item_rows), pd.DataFrame(site_rows), dim_df

# Official population = completers, 10 in-scope sites
comp10 = df[(df["_reached"] == 1) & (df[site_col].isin(TEN_SITES))]
all10  = df[df[site_col].isin(TEN_SITES)]
sc_item, sc_site, sc_dim = official_scores(comp10)
sa_item, sa_site, sa_dim = official_scores(all10)
sc_item.to_csv(os.path.join(DATA, "scores_overall.csv"), index=False)
sc_site.to_csv(os.path.join(DATA, "scores_by_site.csv"), index=False)
sc_dim.to_csv(os.path.join(DATA, "scores_dimension.csv"), index=False)
sa_item.to_csv(os.path.join(DATA, "scores_overall_all.csv"), index=False)
sa_site.to_csv(os.path.join(DATA, "scores_by_site_all.csv"), index=False)
sa_dim.to_csv(os.path.join(DATA, "scores_dimension_all.csv"), index=False)

# ---------------------------------------------------------------------------
# 3. Open-ended -> long, tagged
# ---------------------------------------------------------------------------
def sentiment(text, question_label=None):
    # domain-tuned ensemble; returns (signed score in [-1,1], label). The question
    # prior is keyed by the question LABEL (sentiment.QUESTION_PRIOR).
    return score_sentiment(text, question_label)

records = []
cid = 0
for qcol, qlabel, qdomain in OPEN_ENDED:
    if qcol is None:
        continue
    sub = df[[RESP, site_col, SEG["role"], SEG["hybrid"], SEG["tenure"], "_reached", qcol]].copy()
    sub = sub[sub[qcol].notna()]
    for _, row in sub.iterrows():
        text = str(row[qcol]).strip()
        junk = is_junk(text)
        themes, cats = tag_themes(text)
        sc, slab = sentiment(text, qlabel)
        # If nothing matched but it's a real comment, mark Unclassified
        if not junk and not themes:
            themes, cats = ["Unclassified"], ["Uncategorized"]
        cid += 1
        records.append({
            "comment_id": cid,
            "respondent": row[RESP],
            "reached_end": int(row["_reached"]),
            "site": row[site_col],
            "role": row[SEG["role"]],
            "hybrid": row[SEG["hybrid"]],
            "tenure": row[SEG["tenure"]],
            "source_question": qlabel,
            "question_domain": qdomain,
            "comment": text,
            "char_len": len(text),
            "is_junk": junk,
            "sentiment_score": sc,
            "sentiment_label": "Junk / No comment" if junk else slab,
            "themes": "|".join(themes),
            "categories": "|".join(cats),
            "n_themes": len(themes),
            # editable-override bookkeeping (blank until user edits in app)
            "reviewed": False,
            "themes_override": "",
            "sentiment_override": "",
            "junk_override": "",
            "notes": "",
        })

comments = pd.DataFrame(records)
comments.to_csv(os.path.join(DATA, "comments.csv"), index=False)

# ---------------------------------------------------------------------------
# 3b. Respondent-level overall-agreement scores (feeds key-driver analysis)
#     Four section-summary Likert items map to the four domains.
# ---------------------------------------------------------------------------
AGREE = [
    (find("8. 3. Please indicate your level of agreement"),  "score_work",     "Focused / Private Work"),
    (find("12. 7. Please indicate your level of agreement"), "score_space",    "Physical Environment"),
    (find("16. 11. Please indicate your level of agreement"),"score_services", "Amenities & Services"),
    (find("19. 14. Please indicate your level of agreement"),"score_tech",     "Technology"),
]
resp = df[[RESP, site_col, SEG["role"], SEG["hybrid"], SEG["tenure"], "_reached"]].copy()
resp.columns = ["respondent", "site", "role", "hybrid", "tenure", "reached_end"]
score_cols = []
for col, name, _dom in AGREE:
    if col is not None:
        resp[name] = pd.to_numeric(df[col], errors="coerce")
        score_cols.append(name)
resp["composite"] = resp[score_cols].mean(axis=1, skipna=True)
resp.to_csv(os.path.join(DATA, "respondent_scores.csv"), index=False)
# category -> which section score is its natural outcome
CATEGORY_SCORE = {
    "Focused / Private Work": "score_work", "Collaboration & Meetings": "score_work",
    "Physical Environment": "score_space",
    "Amenities & Services": "score_services", "Workplace Operations": "score_services",
    "Technology": "score_tech", "Policy / Experience": "composite", "Uncategorized": "composite",
}

# ---------------------------------------------------------------------------
# 3c. Emerging-theme discovery — cluster the Unclassified comments
# ---------------------------------------------------------------------------
def discover_emerging(comments_df, k=8):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    unclass = comments_df[(~comments_df["is_junk"]) &
                          (comments_df["themes"] == "Unclassified")]["comment"].dropna()
    unclass = unclass[unclass.str.len() >= 20].reset_index(drop=True)
    if len(unclass) < 40:
        return []
    vec = TfidfVectorizer(stop_words="english", max_features=3000,
                          ngram_range=(1, 2), min_df=3)
    X = vec.fit_transform(unclass)
    k = min(k, max(2, len(unclass) // 25))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    terms = vec.get_feature_names_out()
    dists = km.transform(X)  # distance of each doc to each centroid
    out = []
    for ci in range(k):
        members = (labels == ci)
        size = int(members.sum())
        if size == 0:
            continue
        centroid = km.cluster_centers_[ci]
        keywords = [terms[t] for t in centroid.argsort()[::-1][:8]]
        # representative samples = closest to centroid
        member_idx = [i for i in range(len(unclass)) if members[i]]
        member_idx.sort(key=lambda i: dists[i, ci])
        samples = [unclass.iloc[i] for i in member_idx[:4]]
        out.append({"size": size, "keywords": keywords, "samples": samples})
    return sorted(out, key=lambda c: -c["size"])

emerging = discover_emerging(comments)
with open(os.path.join(DATA, "emerging_themes.json"), "w") as f:
    json.dump(emerging, f, indent=2)

# ---------------------------------------------------------------------------
# 4. Meta
# ---------------------------------------------------------------------------
def opts(col):
    return sorted([str(x) for x in df[col].dropna().unique()])

meta = {
    "n_responses": int(len(df)),
    "n_completers": int((df["_reached"] == 1).sum()),
    "n_comments": int(len(comments)),
    "n_real_comments": int((~comments["is_junk"]).sum()),
    "sites": opts(site_col),
    "roles": opts(SEG["role"]),
    "hybrids": opts(SEG["hybrid"]),
    "tenures": opts(SEG["tenure"]),
    "questions": [q[1] for q in OPEN_ENDED if q[0] is not None],
    "categories": CATEGORIES + ["Uncategorized"],
    "themes": list(THEMES.keys()) + ["Unclassified"],
    "theme_to_category": {**THEME_TO_CATEGORY, "Unclassified": "Uncategorized"},
    "score_cols": score_cols,
    "category_score": CATEGORY_SCORE,
    "n_emerging_clusters": len(emerging),
    "ten_sites": TEN_SITES,
    "dim_weights": DIM_WEIGHTS,
}
with open(os.path.join(DATA, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("=== BUILD COMPLETE ===")
print(f"Responses:        {meta['n_responses']}")
print(f"Comments:         {meta['n_comments']}  (real: {meta['n_real_comments']}, junk: {meta['n_comments']-meta['n_real_comments']})")
print(f"Gap items:        {len(gap_df)}  | by-site cells: {len(gap_site_df)}")
print(f"Sites:            {len(meta['sites'])}")
print("\nSentiment mix:")
print(comments['sentiment_label'].value_counts().to_string())
print("\nTop themes:")
allthemes = comments[~comments['is_junk']]['themes'].str.split('|').explode()
print(allthemes.value_counts().head(15).to_string())
print("\nGap ranking (most under-supported):")
print(gap_df.sort_values('gap', ascending=False)[['item','block','gap','classification']].head(10).to_string(index=False))
print(f"\nRespondent scores: {resp['composite'].notna().sum()} with a composite | "
      f"cols={score_cols}")
print(f"Overall composite mean: {resp['composite'].mean():.2f}")
print(f"Emerging-theme clusters found: {len(emerging)}")
for cl in emerging[:6]:
    print(f"   ({cl['size']:3}) {', '.join(cl['keywords'][:5])}")
