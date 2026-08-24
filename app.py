"""
PulsePoint Open-Ended Response Analyzer
========================================
An interactive tool to review and analyze open-ended survey comments:
  • Gap Analysis .......  why categories are under/over-supported (ratings + comments)
  • Theme & Sentiment ..  trends by theme, sentiment, site, role, tenure
  • Tagging Workspace ..  review & override auto-tags, flag junk, export

Run:  streamlit run app.py
"""
import os, json, re, html as _html, io, zipfile, base64
from collections import defaultdict
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import altair as alt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import action_plan  # Operational Action Plan — "What to Do Next" section (LLM-generated, stored per site)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OVERRIDES = os.path.join(DATA, "overrides.csv")

st.set_page_config(page_title="PulsePoint Comment Analyzer", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
SENT_COLORS = {"Positive": "#2E9E6B", "Neutral / Mixed": "#9AA0A6",
               "Negative": "#D1495B", "Junk / No comment": "#D9DCE1"}
GAP_SCHEME = "redyellowblue"  # high gap (under-supported) = red

# Map each rating item to the comment themes that explain it (for drill-down)
ITEM_THEME_MAP = {
    # Work Types
    "Focused individual work": ["Focus / Concentration", "Noise & Distraction", "Space Availability & Crowding"],
    "Tasks requiring discretion": ["Privacy & Confidentiality"],
    "Planned in-person meetings": ["Meeting & Conference Rooms", "In-Person / Team Presence"],
    "Informal or spontaneous collaboration": ["Collaboration Space", "In-Person / Team Presence"],
    "Working with others on shared": ["Collaboration Space", "Meeting & Conference Rooms"],
    "Virtual collaboration": ["AV / Video Conferencing", "Privacy & Confidentiality", "Noise & Distraction"],
    "Learning, training": ["Collaboration Space", "Meeting & Conference Rooms"],
    # Physical Space
    "spaces that support focused work": ["Focus / Concentration", "Noise & Distraction", "Space Availability & Crowding"],
    "spaces that support discretion": ["Privacy & Confidentiality"],
    "meeting and collaboration spaces": ["Meeting & Conference Rooms", "Collaboration Space"],
    "community spaces": ["Community & Events"],
    "Furniture comfort": ["Furniture & Ergonomics"],
    "personal wellbeing": ["Wellbeing / Wellness Spaces"],
    "Lighting comfort": ["Lighting"],
    "Thermal / temperature": ["Temperature / Thermal"],
    "navigating and finding spaces": ["Navigation / Wayfinding", "Desk Availability & Reservations"],
    # Workplace Services
    "Entry and reception": ["Reception & Entry"],
    "help resolving workplace": ["Support / Help Services", "Hardware (Monitors / Peripherals)", "Connectivity (WiFi / Login / Power)"],
    "community manager": ["Community & Events", "Support / Help Services"],
    "food and beverage": ["Food & Beverage", "Cost / Pricing"],
    "workplace support services": ["Support / Help Services", "Wellbeing / Wellness Spaces"],
    "Cleanliness and upkeep": ["Cleanliness & Upkeep"],
    "building operations": ["Building Operations", "Parking & Commute"],
    "Communication about workplace": ["Communication & Updates"],
}

def themes_for_item(item):
    for key, themes in ITEM_THEME_MAP.items():
        if key.lower() in str(item).lower():
            return themes
    return []

# ---------------------------------------------------------------------------
# Data loading (with overrides merged in)
# ---------------------------------------------------------------------------
@st.cache_data
def load_all():
    comments = pd.read_csv(os.path.join(DATA, "comments.csv"))
    with open(os.path.join(DATA, "meta.json")) as f:
        meta = json.load(f)
    gaps = {
        "comp": pd.read_csv(os.path.join(DATA, "gap_overall.csv")),
        "comp_site": pd.read_csv(os.path.join(DATA, "gap_by_site.csv")),
    }
    for key, fn in [("all", "gap_overall_all.csv"), ("all_site", "gap_by_site_all.csv")]:
        p = os.path.join(DATA, fn)
        gaps[key] = pd.read_csv(p) if os.path.exists(p) else gaps["comp" if key == "all" else "comp_site"]
    if "reached_end" not in comments.columns:
        comments["reached_end"] = 1
    # ensure override columns
    for col, default in [("themes_override",""),("sentiment_override",""),
                         ("junk_override",""),("notes",""),("reviewed",False)]:
        if col not in comments.columns:
            comments[col] = default
    comments["themes"] = comments["themes"].fillna("")
    # merge saved overrides
    if os.path.exists(OVERRIDES):
        ov = pd.read_csv(OVERRIDES)
        comments = comments.drop(columns=[c for c in
                    ["themes_override","sentiment_override","junk_override","notes","reviewed"]
                    if c in comments.columns])
        comments = comments.merge(ov, on="comment_id", how="left")
    for col, default in [("themes_override",""),("sentiment_override",""),
                         ("junk_override",""),("notes",""),("reviewed",False)]:
        if col not in comments.columns:
            comments[col] = default
        comments[col] = comments[col].fillna(default)
    # effective values
    def eff_junk(r):
        o = str(r["junk_override"]).strip().lower()
        if o in ("true","yes","1"): return True
        if o in ("false","no","0"): return False
        return bool(r["is_junk"])
    comments["eff_junk"] = comments.apply(eff_junk, axis=1)
    comments["eff_sentiment"] = np.where(
        comments["sentiment_override"].astype(str).str.strip() != "",
        comments["sentiment_override"], comments["sentiment_label"])
    comments["eff_sentiment"] = np.where(comments["eff_junk"], "Junk / No comment",
                                         comments["eff_sentiment"])
    comments["eff_themes"] = np.where(
        comments["themes_override"].astype(str).str.strip() != "",
        comments["themes_override"], comments["themes"])
    return comments, meta, gaps

comments_full, meta, GAPS = load_all()

@st.cache_data
def load_scores():
    """Official Tableau-spec scores (Friction/Composite/RYG) for both populations."""
    numcols = ["n", "mean_importance", "mean_support", "importance_100", "support_100",
               "composite", "friction", "friction_pct", "weight"]
    def rd(fn):
        p = os.path.join(DATA, fn)
        if not os.path.exists(p):
            return pd.DataFrame()
        d = pd.read_csv(p)
        for c in numcols:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        return d
    return {
        "comp": rd("scores_overall.csv"), "comp_site": rd("scores_by_site.csv"), "comp_dim": rd("scores_dimension.csv"),
        "all": rd("scores_overall_all.csv"), "all_site": rd("scores_by_site_all.csv"), "all_dim": rd("scores_dimension_all.csv"),
    }

SCORES = load_scores()
THEME_TO_CAT = meta["theme_to_category"]

# ---------------------------------------------------------------------------
# Friction (Tableau spec) attributed to themes / categories — feeds priorities
# ---------------------------------------------------------------------------
def _theme_friction_from(sdf):
    tmp = defaultdict(list)
    if sdf is None or sdf.empty:
        return {}
    for _, r in sdf.iterrows():
        fr = r.get("friction")
        if pd.isna(fr):
            continue
        for th in themes_for_item(r["item"]):
            tmp[th].append(float(fr))
    return {th: float(np.mean(v)) for th, v in tmp.items()}

def theme_friction_map(site=None):
    """Theme->Friction (0-100). Site-specific when a single site is selected."""
    if site and not scores_site.empty and site in set(scores_site["site"]):
        m = _theme_friction_from(scores_site[scores_site["site"] == site])
        return {**THEME_FRICTION_ALL, **m}
    return THEME_FRICTION_ALL

def category_friction_map(tfric):
    tmp = defaultdict(list)
    for th, g in tfric.items():
        tmp[THEME_TO_CAT.get(th, "Uncategorized")].append(g)
    return {c: float(np.mean(v)) for c, v in tmp.items()}

def explode_level(df, level):
    """Explode eff_themes to one row per theme; roll up to category if requested."""
    t = df.assign(_k=df["eff_themes"].str.split("|")).explode("_k")
    t["_k"] = t["_k"].str.strip()
    t = t[(t["_k"] != "") & (t["_k"] != "Unclassified")]
    if level == "Categories":
        t["_k"] = t["_k"].map(lambda x: THEME_TO_CAT.get(x, "Uncategorized"))
        t = t[t["_k"] != "Uncategorized"]
    # reset index: explode leaves duplicate labels, which break crosstab/alignment
    return t.rename(columns={"_k": "key"}).reset_index(drop=True)

def build_priority(df, level, tfric):
    real = df[~df["eff_junk"]]
    if real.empty:
        return pd.DataFrame()
    ex = explode_level(real, level)
    fmap = category_friction_map(tfric) if level == "Categories" else tfric
    rows = []
    for k, gg in ex.groupby("key"):
        rows.append((k, gg["comment_id"].nunique(),
                     (gg["eff_sentiment"] == "Negative").mean(), fmap.get(k, np.nan)))
    P = pd.DataFrame(rows, columns=["key", "volume", "neg_share", "friction"])
    P["vol_norm"] = P["volume"] / P["volume"].max()
    fr = P["friction"].clip(lower=0)
    denom = fr.max() if fr.max() and fr.max() > 0 else 1.0
    fn = fr / denom
    P["fric_norm"] = fn.fillna(fn.median() if fn.notna().any() else 0.0)
    P["Priority"] = (0.40 * P["vol_norm"] + 0.35 * P["neg_share"] + 0.25 * P["fric_norm"]) * 100
    return P.sort_values("Priority", ascending=False)

def exemplars(df, key, level, n=3):
    real = df[~df["eff_junk"]]
    if level == "Categories":
        mask = real["eff_themes"].apply(
            lambda s: any(THEME_TO_CAT.get(t.strip()) == key for t in str(s).split("|")))
    else:
        mask = real["eff_themes"].str.contains(re.escape(key), na=False)
    sub = real[mask & (real["eff_sentiment"] == "Negative")]
    sub = sub[(sub["char_len"] >= 40) & (sub["char_len"] <= 280)]
    return sub.sort_values("sentiment_score").head(n)

# ---------------------------------------------------------------------------
# Extra data: respondent-level scores (key drivers) + emerging clusters
# ---------------------------------------------------------------------------
@st.cache_data
def load_extra():
    try:
        rs = pd.read_csv(os.path.join(DATA, "respondent_scores.csv"))
    except Exception:
        rs = pd.DataFrame()
    try:
        with open(os.path.join(DATA, "emerging_themes.json")) as f:
            emg = json.load(f)
    except Exception:
        emg = []
    return rs, emg

resp_scores_full, emerging = load_extra()
SCORE_LABELS = {"Overall (all sections)": "composite", "Workspace support": "score_work",
                "Physical space": "score_space", "Services": "score_services",
                "Technology": "score_tech"}

def build_hotspots(comments_df, dim="site", min_n=40, min_lift=1.15, alpha=0.05):
    """Theme rates in a group significantly above the rest (two-proportion z-test)."""
    from scipy.stats import norm
    real = comments_df[~comments_df["eff_junk"]]
    ex = explode_level(real, "Themes")
    org_tot = real["comment_id"].nunique()
    org_theme = ex.groupby("key")["comment_id"].nunique()
    grp_tot = real.groupby(dim)["comment_id"].nunique()
    rows = []
    for (g, th), gg in ex.groupby([dim, "key"]):
        n_g = int(grp_tot.get(g, 0))
        if n_g < min_n:
            continue
        x_g = gg["comment_id"].nunique()
        x_r = int(org_theme.get(th, 0)) - x_g
        n_r = org_tot - n_g
        if n_r <= 0:
            continue
        p_g, p_r = x_g / n_g, x_r / n_r
        org_rate = int(org_theme.get(th, 0)) / org_tot
        if p_g <= org_rate:
            continue
        pool = (x_g + x_r) / (n_g + n_r)
        se = np.sqrt(pool * (1 - pool) * (1 / n_g + 1 / n_r))
        z = (p_g - p_r) / se if se > 0 else 0.0
        pval = 1 - norm.cdf(z)
        lift = p_g / org_rate if org_rate > 0 else np.nan
        if lift >= min_lift and pval < alpha:
            rows.append((g, th, n_g, x_g, round(p_g * 100, 1),
                         round(org_rate * 100, 1), round(lift, 2), round(pval, 4)))
    return pd.DataFrame(rows, columns=[dim, "theme", "group_n", "mentions",
                        "group_rate", "org_rate", "lift", "p"]).sort_values("lift", ascending=False)

def _theme_sets(comments_df):
    real = comments_df[~comments_df["eff_junk"]]
    return real.groupby("respondent")["eff_themes"].apply(
        lambda s: set(t for t in "|".join(s).split("|") if t and t != "Unclassified"))

def build_driver(comments_df, rs, score_col="composite"):
    from scipy import stats
    if rs.empty or score_col not in rs.columns:
        return pd.DataFrame()
    d = rs.set_index("respondent").join(_theme_sets(comments_df).rename("ts"))
    d = d.dropna(subset=[score_col])
    N = len(d)
    themes = sorted({t for S in d["ts"].dropna() for t in S})
    rows = []
    for th in themes:
        has = d["ts"].apply(lambda S: isinstance(S, set) and th in S)
        a = d.loc[has, score_col].dropna()
        b = d.loc[~has, score_col].dropna()
        if len(a) < 20:
            continue
        delta = a.mean() - b.mean()
        try:
            _, p = stats.ttest_ind(a, b, equal_var=False)
        except Exception:
            p = np.nan
        rows.append((th, len(a), round(a.mean(), 2), round(b.mean(), 2),
                     round(delta, 2), p, round(delta * len(a) / N, 3)))
    return pd.DataFrame(rows, columns=["theme", "n", "sat_with", "sat_without",
                        "delta", "p", "impact"]).sort_values("impact")

def _norm_text(t):
    return re.sub(r"\s+", " ", str(t).strip().lower())[:90]

def select_quotes(real, key, level, kind, n, seen_ids, seen_texts):
    """Pick deduped, representative quotes for a theme/category.
    kind: 'critical' (most negative), 'representative' (typical negative), 'positive'."""
    if level == "Categories":
        mask = real["eff_themes"].apply(
            lambda s: any(THEME_TO_CAT.get(t.strip()) == key for t in str(s).split("|")))
    else:
        mask = real["eff_themes"].str.contains(re.escape(key), na=False)
    pool = real[mask].copy()
    pool = pool[(pool["char_len"] >= 45) & (pool["char_len"] <= 300)]
    if kind == "positive":
        pool = pool[pool["eff_sentiment"] == "Positive"].sort_values("sentiment_score", ascending=False)
    else:
        pool = pool[pool["eff_sentiment"] == "Negative"]
        pool["_nth"] = pool["eff_themes"].str.count(r"\|") + 1  # fewer themes = more focused
        if kind == "representative":
            med = pool["sentiment_score"].median() if len(pool) else 0
            pool["_dist"] = (pool["sentiment_score"] - med).abs()
            pool = pool.sort_values(["_nth", "_dist", "char_len"])
        else:  # critical
            pool = pool.sort_values(["sentiment_score", "_nth"])
    out = []
    for _, r in pool.iterrows():
        nt = _norm_text(r["comment"])
        if r["comment_id"] in seen_ids or nt in seen_texts:
            continue
        seen_ids.add(r["comment_id"]); seen_texts.add(nt)
        out.append(r)
        if len(out) >= n:
            break
    return out

def _b64png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# RYG colors matching the official Tableau legend
RYG_HEX = {"Under Supported": "#C0392B", "Target": "#2E7D32",
           "Over Supported": "#E67E22", "Agreement-only": "#9AA0A6"}
DIM_ORDER = {"Work Activity": 0, "Physical Feature": 1, "Service Level": 2, "Digital & Tech": 3}
# each dimension's single summary "agreement" question (Q8/Q12/Q16/Q19) -> respondent score column
DIM_TO_SCORECOL = {"Work Activity": "score_work", "Physical Feature": "score_space",
                   "Service Level": "score_services", "Digital & Tech": "score_tech"}

def ryg_of(gap5):
    """RYG status from Support − Importance (1–5), per the Tableau spec thresholds."""
    if pd.isna(gap5):
        return "Agreement-only"
    if gap5 < -0.049:
        return "Under Supported"
    if gap5 > 1.5:
        return "Over Supported"
    return "Target"

# --- Report-card grading (Composite 0–100 support score -> letter grade) -----
GRADE_BANDS = [(80, "A", "Excellent"), (72, "B", "Good"), (64, "C", "Fair"),
               (56, "D", "Needs Work"), (0, "F", "Poor")]
GRADE_HEX = {"A": "#1E7D32", "B": "#5A9E3A", "C": "#E0A400", "D": "#E67E22", "F": "#C0392B"}
# lay-language status words for the RYG classification (kept alongside the numbers)
STATUS_WORD = {"Under Supported": "Needs attention", "Target": "On track",
               "Over Supported": "Exceeds needs", "Agreement-only": "Agreement-based"}
# Digital & Tech is agreement-only (no Importance/Gap). Give it a real status from the
# 1–5 agreement level so a 4.0 vs a 3.6 is meaningful (net-agree = strong).
TECH_BANDS = [(4.0, "Strong", "#2E7D32"), (3.5, "Moderate", "#E0A400"), (0.0, "Needs attention", "#C0392B")]

def tech_status(mean_support_1to5):
    """(label, hex) for an agreement-based score on the 1–5 scale."""
    if pd.isna(mean_support_1to5):
        return ("—", "#999999")
    for lo, lab, hx in TECH_BANDS:
        if mean_support_1to5 >= lo:
            return (lab, hx)
    return ("Needs attention", "#C0392B")

def grade_of(score):
    """Return (letter, word, hex) for a 0–100 composite score."""
    if pd.isna(score):
        return ("—", "", "#999999")
    for lo, letter, word in GRADE_BANDS:
        if score >= lo:
            return (letter, word, GRADE_HEX[letter])
    return ("F", "Poor", GRADE_HEX["F"])

def vs_mark(site_val, org_val, thresh=2.0):
    """Professional vs-average indicator (▲/▼/~) with color — no emoji."""
    if pd.isna(site_val) or pd.isna(org_val):
        return ""
    d = site_val - org_val
    if d >= thresh:
        return f"<span style='color:#1E7D32'>&#9650; +{d:.0f}</span>"      # ▲
    if d <= -thresh:
        return f"<span style='color:#C0392B'>&#9660; {d:.0f}</span>"       # ▼
    return "<span style='color:#888'>&#8776; on par</span>"                # ≈

def _f1(v):
    """One-decimal, or em-dash for missing (agreement-only items carry no Importance/Gap)."""
    return "—" if pd.isna(v) else f"{v:.1f}"

def _gp(v):
    """Signed one-decimal gap, or em-dash for agreement-only items."""
    return "—" if pd.isna(v) else f"{v:+.1f}"

# --- Relative standing among the in-scope sites (replaces absolute A–F letter grades) ---
TIER_HEX = {"Top third": "#1E7D32", "Middle third": "#E0A400",
            "Bottom third": "#C0392B", "—": "#999999"}

def rank_tier(value, dist):
    """Rank (1 = best) + distribution tercile for a value within the 10-site spread.
    Returns (rank, n_sites, tier_label, tier_hex, percentile). Descriptive, not an
    invented cutoff — the tier is defined by where the site falls among its peers."""
    vals = sorted([float(v) for v in dist if pd.notna(v)], reverse=True)
    n = len(vals)
    if n == 0 or pd.isna(value):
        return (None, n, "—", TIER_HEX["—"], None)
    rank = sum(1 for v in vals if v > value) + 1
    p33, p67 = np.percentile(vals, [100 / 3, 200 / 3])
    tier = "Top third" if value >= p67 else ("Bottom third" if value <= p33 else "Middle third")
    pctl = round(100 * sum(1 for v in vals if v < value) / n)
    return (rank, n, tier, TIER_HEX[tier], pctl)

@st.cache_data(show_spinner=False)
def site_standings(completers_flag):
    """Per-site overall Index Score (unweighted mean of the four dimension composites,
    0–100 — this is what matches the Tableau deck) + per-dimension composites, for ranking."""
    ss = scores_site
    res = {"overall": {}, "dims": {}}
    if ss is None or ss.empty:
        return res
    dimc = (ss.dropna(subset=["support_100"])
              .groupby(["site", "dimension"])["support_100"].mean().reset_index())
    for dim, g in dimc.groupby("dimension"):
        res["dims"][dim] = dict(zip(g["site"], g["support_100"]))
    for site, g in dimc.groupby("site"):
        dd = dict(zip(g["dimension"], g["support_100"]))
        if dd:
            res["overall"][site] = float(np.mean(list(dd.values())))  # unweighted = deck method
    return res

def agreement_sig(site, col, ten_sites):
    """Welch's t-test of a site's 1–5 summary-agreement scores vs the OTHER in-scope
    sites. Returns site/rest means, delta, p, a plain-language verdict, and a color.
    This is the defensible 'raw number + a statistical thing' for the overall grade."""
    if resp_scores is None or resp_scores.empty or col not in resp_scores.columns:
        return None
    insc = resp_scores[resp_scores["site"].isin(ten_sites)]
    a = pd.to_numeric(insc.loc[insc["site"] == site, col], errors="coerce").dropna()
    b = pd.to_numeric(insc.loc[insc["site"] != site, col], errors="coerce").dropna()
    sm = float(a.mean()) if len(a) else np.nan
    om = float(b.mean()) if len(b) else np.nan
    if len(a) < 10 or len(b) < 10:
        return {"n": int(len(a)), "site_mean": sm, "org_mean": om, "delta": (sm - om),
                "p": np.nan, "verdict": "too few responses to test", "hex": "#999999"}
    from scipy.stats import ttest_ind
    _t, p = ttest_ind(a, b, equal_var=False)
    delta = sm - om
    if pd.notna(p) and p < 0.05 and delta > 0:
        verdict, hexc = "above company (significant)", "#1E7D32"
    elif pd.notna(p) and p < 0.05 and delta < 0:
        verdict, hexc = "below company (significant)", "#C0392B"
    else:
        verdict, hexc = "in line with company", "#6b7280"
    return {"n": int(len(a)), "site_mean": sm, "org_mean": om, "delta": float(delta),
            "p": float(p), "verdict": verdict, "hex": hexc}

@st.cache_data(show_spinner=False)
def site_report_data(site, completers_flag):
    """All computed pieces for a site's report (cached per site+filter)."""
    sc = comments[comments["site"] == site]
    real = sc[~sc["eff_junk"]].copy()
    n = len(real)
    data = {"site": site, "n": n}
    if n == 0:
        return data
    sent = real["eff_sentiment"].value_counts()
    data["neg"] = float((real["eff_sentiment"] == "Negative").mean() * 100)
    data["pos"] = float((real["eff_sentiment"] == "Positive").mean() * 100)
    data["neu"] = 100 - data["neg"] - data["pos"]
    # priorities (categories) + top themes
    data["P"] = build_priority(sc, "Categories", theme_friction_map(site)).head(5)
    tex = explode_level(real, "Themes")
    tcounts = tex["key"].value_counts()
    data["top_themes"] = tcounts.head(8)
    # theme sentiment for strengths/fixes
    tsent = tex.groupby("key")["eff_sentiment"].apply(lambda s: (s == "Negative").mean())
    tvol = tcounts
    fixes = [(t, int(tvol[t]), float(tsent[t])) for t in tvol.index
             if tvol[t] >= 5 and tsent.get(t, 0) >= 0.45][:5]
    strengths = [(t, int(tvol[t]), float(1 - tsent.get(t, 1))) for t in tvol.index
                 if tvol[t] >= 5 and tsent.get(t, 1) <= 0.30][:5]
    data["fixes"], data["strengths"] = fixes, strengths
    # official scores (Tableau spec) — only exist for the 10 in-scope sites
    ss = scores_site[scores_site["site"] == site].dropna(subset=["friction"]) if not scores_site.empty else pd.DataFrame()
    imp_items = ss[ss["mean_importance"].notna()] if len(ss) else pd.DataFrame()  # importance-based dims only
    data["under"] = imp_items.sort_values("friction", ascending=False).head(4) if len(imp_items) else pd.DataFrame()
    data["over"] = ss[ss["ryg"] == "Over Supported"].head(2) if len(ss) else pd.DataFrame()
    data["in_scope"] = len(ss) > 0
    # Importance vs Support DETAIL (1–5, Gap = Support − Importance) — the Tableau view.
    # Digital & Tech is agreement-only (no Importance rating): included here with its
    # Support/agreement score, and Gap/Status shown as "Agreement-based" rather than invented.
    det = ss.copy() if len(ss) else pd.DataFrame()
    if len(det):
        det["gap5"] = (det["mean_support"] - det["mean_importance"]).round(1) + 0.0  # NaN for Tech
        det["_o"] = det["dimension"].map(DIM_ORDER).fillna(9)
        # sort within a dimension by gap (importance dims) or by support (agreement-only Tech)
        det["_k"] = det["gap5"].where(det["gap5"].notna(), det["mean_support"] - 5.0)
        det = det.sort_values(["_o", "_k"], ascending=[True, False])
        dsum = det.groupby("dimension", sort=False).agg(
            mean_importance=("mean_importance", "mean"),
            mean_support=("mean_support", "mean")).reset_index()
        dsum["gap5"] = (dsum["mean_support"] - dsum["mean_importance"]).round(1) + 0.0
        dsum["_o"] = dsum["dimension"].map(DIM_ORDER).fillna(9)
        dsum = dsum.sort_values("_o")
        data["detail"], data["dim_summary"] = det, dsum
    else:
        data["detail"], data["dim_summary"] = pd.DataFrame(), pd.DataFrame()

    # Report card: dimension composites (0–100) -> grades, vs org average
    data["participation"] = int((resp_scores["site"] == site).sum()) if not resp_scores.empty else data["n"]
    _insc = resp_scores[resp_scores["site"].isin(meta.get("ten_sites", []))] if not resp_scores.empty else pd.DataFrame()
    data["all_participants"] = int(len(_insc))
    # full comment log for the appendix (all real comments at this site)
    data["real_df"] = real[["respondent", "comment", "eff_sentiment", "eff_themes", "role", "tenure"]].reset_index(drop=True)
    ten = meta.get("ten_sites", [])
    standings = site_standings(completers_flag)
    if len(ss):
        dcomp = ss.groupby("dimension")["support_100"].mean()
        orgd = scores_dim.set_index("dimension")["composite"] if not scores_dim.empty else pd.Series(dtype=float)
        # Overall PulsePoint Experience Index Score = unweighted mean of the four dimension
        # composites (0–100). This is the calculation that matches the Tableau deck.
        site_overall = float(dcomp.mean())
        o_rank, o_n, o_tier, o_hex, o_pctl = rank_tier(site_overall, list(standings["overall"].values()))
        areas = []
        for dm in ["Work Activity", "Physical Feature", "Service Level", "Digital & Tech"]:
            if dm in dcomp.index:
                a_rank, a_n, a_tier, a_hex, _ap = rank_tier(
                    float(dcomp[dm]), list(standings["dims"].get(dm, {}).values()))
                areas.append({"dim": dm, "score": float(dcomp[dm]), "org": float(orgd.get(dm, np.nan)),
                              "rank": a_rank, "n": a_n, "tier": a_tier, "hex": a_hex})
        ovals = [v for v in standings["overall"].values() if pd.notna(v)]
        data["card"] = {"overall": site_overall,
                        "org_overall": float(orgd.get("OVERALL (equal)", np.nan)),
                        "areas": areas, "rank": o_rank, "n_sites": o_n, "tier": o_tier,
                        "tier_hex": o_hex, "pctl": o_pctl,
                        "dist_min": (min(ovals) if ovals else np.nan),
                        "dist_max": (max(ovals) if ovals else np.nan)}
    else:
        data["card"] = None
    # Strengths — best-supported items (mirror of the Friction chart), importance dims
    imp_ss = ss[ss["mean_importance"].notna()] if len(ss) else pd.DataFrame()
    data["top_support"] = (imp_ss.sort_values("support_100", ascending=False).head(4)
                           if len(imp_ss) else pd.DataFrame())
    # Demographics — who responded at this site (counts + % of site)
    rs_site_all = resp_scores[resp_scores["site"] == site] if not resp_scores.empty else pd.DataFrame()
    demo = {}
    for lab, col in [("Role", "role"), ("Tenure", "tenure"), ("Hybrid category", "hybrid")]:
        if len(rs_site_all) and col in rs_site_all.columns:
            vc = rs_site_all[col].dropna().astype(str).value_counts()
            if len(vc):
                demo[lab] = vc
    data["demographics"] = demo
    data["demo_total"] = int(len(rs_site_all))
    # benchmark vs org: theme rate lift
    org_real = comments[~comments["eff_junk"]]
    org_tot = len(org_real)
    org_rate = {t: (org_real["eff_themes"].str.contains(re.escape(t), na=False)).sum() / org_tot
                for t in tvol.index[:8]}
    data["bench"] = [(t, tvol[t] / n, org_rate.get(t, 0)) for t in tvol.index[:6]]
    # satisfaction scores vs org
    rs_site = resp_scores[resp_scores["site"] == site]
    scoremap = {"Workspace": "score_work", "Physical space": "score_space",
                "Services": "score_services", "Technology": "score_tech", "Overall": "composite"}
    data["scores"] = []
    for lab, col in scoremap.items():
        if col in resp_scores.columns:
            sv = rs_site[col].mean(); ov = resp_scores[col].mean()
            if pd.notna(sv):
                data["scores"].append((lab, float(sv), float(ov)))
    # --- Employee sentiment: the survey's own summary agreement questions (1–5) ------
    #     Each dimension has one summary Likert item (Q8/Q12/Q16/Q19). We surface them
    #     here with a Welch t-test vs the other in-scope sites (the "statistical thing").
    # Q16 (services block) and Q19 (technology block) carry identical wording in the export;
    # they are distinguished here by survey position, per confirmation to relabel by position.
    AGREE_Q = {
        "Work Activity": "“My workplace enables me to work productively” (Q8)",
        "Physical Feature": "“I feel comfortable and safe while working at the PulsePoint” (Q12)",
        "Service Level": "Overall ease of using the PulsePoint — Services (Q16)",
        "Digital & Tech": "Overall ease of using the PulsePoint — Technology (Q19)"}
    agreement = []
    for dm in ["Work Activity", "Physical Feature", "Service Level", "Digital & Tech"]:
        sig = agreement_sig(site, DIM_TO_SCORECOL[dm], ten)
        if sig is not None:
            agreement.append({"dim": dm, "q": AGREE_Q[dm], **sig})
    data["agreement"] = agreement
    data["sentiment_overall"] = agreement_sig(site, "composite", ten)
    # attach each dimension's summary-agreement score to the Dimension summary table
    if len(data["dim_summary"]):
        amap = {a["dim"]: a for a in agreement}
        ds = data["dim_summary"]
        ds["agree"] = ds["dimension"].map(lambda dm: amap.get(dm, {}).get("site_mean", np.nan))
        ds["agree_verdict"] = ds["dimension"].map(lambda dm: amap.get(dm, {}).get("verdict", ""))
        ds["agree_hex"] = ds["dimension"].map(lambda dm: amap.get(dm, {}).get("hex", "#888"))
    # verbatims: top themes -> representative negative quote each; + positives
    seen_ids, seen_texts = set(), set()
    quotes = []
    for t in tvol.index[:6]:
        qs = select_quotes(real, t, "Themes", "representative", 1, seen_ids, seen_texts)
        if qs:
            r = qs[0]
            same = int(((tex["key"] == t) & (tex["eff_sentiment"] == r["eff_sentiment"])).sum())
            quotes.append({"theme": t, "text": str(r["comment"]), "role": str(r["role"]),
                           "tenure": str(r["tenure"]), "sentiment": r["eff_sentiment"], "represents": same})
        if len(quotes) >= 5:
            break
    data["quotes"] = quotes
    pos_quotes = []
    for t in [s[0] for s in strengths] or list(tvol.index[:4]):
        qs = select_quotes(real, t, "Themes", "positive", 1, seen_ids, seen_texts)
        if qs:
            r = qs[0]
            pos_quotes.append({"theme": t, "text": str(r["comment"]), "role": str(r["role"]),
                               "tenure": str(r["tenure"])})
        if len(pos_quotes) >= 2:
            break
    data["pos_quotes"] = pos_quotes
    data["charts"] = _site_charts(data)
    return data

SENT_HEX = {"Positive": "#2E9E6B", "Neutral / Mixed": "#9AA0A6", "Negative": "#D1495B"}

DIM_ABBR = {"Work Activity": "Work Activity", "Physical Feature": "Physical Feature",
            "Service Level": "Service Level", "Digital & Tech": "Digital & Tech"}

def _wrap(s, w=48):
    import textwrap
    return "\n".join(textwrap.wrap(str(s), w)) or str(s)

def _score_bar(rows, valcol, color, title, xlabel, dark, pctcol=None, xmax=None):
    """Shared horizontal-bar renderer for Friction (red) and Strengths (green) — larger,
    wrapped labels, consistent formatting per the reviewers' request."""
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    vals = [float(r[valcol]) for _, r in rows.iterrows()]
    ax.barh(range(len(rows)), vals, color=color, height=0.62)
    labels = [_wrap(f"[{DIM_ABBR.get(r['dimension'], r['dimension'])}] {r['item']}", 50)
              for _, r in rows.iterrows()]
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels, fontsize=8)
    vmax = xmax or (max(vals) if vals else 1)
    for i, (_, r) in enumerate(rows.iterrows()):
        tail = ""
        if pctcol and pd.notna(r.get(pctcol)):
            tail = f" · {r[pctcol]:.0f}% feel it"
        ax.text(float(r[valcol]) + vmax * 0.02, i, f"{float(r[valcol]):.0f}/100{tail}",
                va="center", fontsize=8.5, color=dark, weight="bold")
    ax.set_xlim(0, vmax * (1.5 if pctcol else 1.18))
    ax.set_xlabel(xlabel, fontsize=9); ax.tick_params(axis="x", labelsize=8)
    ax.set_title(title, fontsize=11.5, weight="bold", color="#2b3a67", loc="left")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    fig.tight_layout()
    return _b64png(fig)

def _site_charts(d):
    """Return dict of base64 PNG charts for a site report."""
    charts = {}
    # donut — comment sentiment (titled)
    fig, ax = plt.subplots(figsize=(2.7, 2.9))
    vals = [d["pos"], d["neu"], d["neg"]]
    ax.pie(vals, colors=[SENT_HEX["Positive"], SENT_HEX["Neutral / Mixed"], SENT_HEX["Negative"]],
           startangle=90, wedgeprops=dict(width=0.42))
    ax.text(0, 0, f"{d['neg']:.0f}%\nneg", ha="center", va="center", fontsize=11, weight="bold")
    ax.set(aspect="equal")
    ax.set_title("Comment sentiment\n(green +  ·  grey ~  ·  red −)", fontsize=9, weight="bold", color="#2b3a67")
    charts["donut"] = _b64png(fig)
    # top themes bar — % of this site's comments + comment count on each bar
    tt = d["top_themes"].head(7)[::-1]
    n_real = max(1, int(d.get("n", 1)))
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.barh(range(len(tt)), tt.values, color="#4C78A8", height=0.62)
    ax.set_yticks(range(len(tt))); ax.set_yticklabels([_wrap(str(t), 24) for t in tt.index], fontsize=7.5)
    tmax = float(tt.max()) if len(tt) else 1
    for i, (name, c) in enumerate(tt.items()):
        ax.text(c + tmax * 0.02, i, f"{c / n_real * 100:.0f}% | {int(c)} comments",
                va="center", fontsize=7.5, color="#33475b", weight="bold")
    ax.set_xlim(0, tmax * 1.42)
    ax.set_xlabel("comment mentions", fontsize=8); ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Top comment themes (% of this site's comments)", fontsize=10.5,
                 weight="bold", color="#2b3a67", loc="left")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    fig.tight_layout(); charts["themes"] = _b64png(fig)
    # Friction (red) + Strengths (green) — enlarged, wrapped, consistent formatting
    if len(d["under"]):
        charts["gap"] = _score_bar(d["under"].sort_values("friction"), "friction", "#D1495B",
                                   "Where support falls short — highest-Friction items",
                                   "Friction (0–100, higher = worse)", "#7a1f2b",
                                   pctcol="friction_pct", xmax=100)
    if len(d.get("top_support", [])):
        charts["strength"] = _score_bar(d["top_support"].sort_values("support_100"), "support_100", "#2E7D32",
                                        "PulsePoint strengths — best-supported items",
                                        "Support (0–100, higher = better)", "#1c5024", xmax=100)
    # Demographics visual — % of site by group (role / tenure / hybrid)
    demo = d.get("demographics") or {}; dt = d.get("demo_total") or 0
    if demo and dt:
        keys = list(demo.keys())
        fig, axes = plt.subplots(1, len(keys), figsize=(3.05 * len(keys), 2.7))
        if len(keys) == 1:
            axes = [axes]
        for ax, lab in zip(axes, keys):
            vc = demo[lab].head(6)[::-1]; pct = vc / dt * 100
            ax.barh(range(len(vc)), pct.values, color="#5B7DB1", height=0.62)
            ax.set_yticks(range(len(vc))); ax.set_yticklabels([_wrap(str(k), 16) for k in vc.index], fontsize=7)
            for i, v in enumerate(pct.values):
                ax.text(v + 1.5, i, f"{v:.0f}%", va="center", fontsize=7.5, color="#33475b")
            ax.set_xlim(0, 112); ax.set_title(lab, fontsize=9.5, weight="bold", color="#2b3a67")
            ax.tick_params(axis="x", labelsize=7)
            for s in ["top", "right"]: ax.spines[s].set_visible(False)
        fig.tight_layout(); charts["demo"] = _b64png(fig)
    return charts

def html_to_pdf(html):
    """Render report HTML to PDF so the PDF export is identical to the HTML report.
    Tries browsers in order — the system Chrome, then Edge (no download, works on
    locked-down/corporate machines), then Playwright's bundled Chromium — and finally
    falls back to WeasyPrint."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = None
            for _launch in (
                lambda: pw.chromium.launch(channel="chrome", args=["--no-sandbox"]),
                lambda: pw.chromium.launch(channel="msedge", args=["--no-sandbox"]),
                lambda: pw.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"]),
            ):
                try:
                    browser = _launch(); break
                except Exception:
                    continue
            if browser is not None:
                page = browser.new_page()
                page.set_content(html, wait_until="load")
                pdf = page.pdf(format="Letter", print_background=True, prefer_css_page_size=True,
                               margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
                browser.close()
                return pdf
    except Exception:
        pass
    try:
        from weasyprint import HTML as _WHTML
        return _WHTML(string=html).write_pdf()
    except Exception as exc:
        raise RuntimeError(
            "PDF export needs a browser. Easiest: download the HTML report and use your browser's "
            "Print → Save as PDF (identical output). To enable the in-app button, install Chrome or "
            f"Edge, or run `python -m playwright install chromium`. (renderer error: {exc})")

def build_site_report(site):
    """Self-contained HTML one-pager for one site."""
    d = site_report_data(site, completers_only)
    if d["n"] == 0:
        return f"<html><body style='font-family:sans-serif'><h2>{_html.escape(site)}</h2><p>No comments for this site.</p></body></html>"
    charts = d["charts"]
    # Operational Action Plan — "What to Do Next": rendered from the stored per-site
    # plan (empty string until one is generated, so exports stay clean).
    action_plan_html = action_plan.render_action_plan_html(action_plan.load_plan(site))
    P, tops = d["P"], d["P"]["key"].tolist()
    _ncommenters = int(d["real_df"]["respondent"].nunique()) if d.get("real_df") is not None and len(d.get("real_df", [])) else 0
    narrative = (f"We received <b>{d['n']:,} substantive comments</b>"
                 + (f" from {_ncommenters:,} associates" if _ncommenters else "")
                 + f" at <b>{_html.escape(site)}</b> (associates could answer up to five open-ended questions). "
                 f"Overall comment sentiment is <b>{d['neg']:.0f}% negative</b> and {d['pos']:.0f}% positive. ")
    if tops:
        narrative += (f"The top area to address is <b>{_html.escape(tops[0])}</b> "
                      f"({int(P.iloc[0]['neg_share']*100)}% negative across {int(P.iloc[0]['volume'])} comments)")
        if len(tops) > 1: narrative += f", then {_html.escape(tops[1])}"
        if len(tops) > 2: narrative += f" and {_html.escape(tops[2])}"
        narrative += ". "
    if len(d["under"]):
        u0 = d["under"].iloc[0]
        narrative += (f"The highest-Friction item is <b>{_html.escape(str(u0['item']))}</b> "
                      f"(Friction {u0['friction']:.0f}/100, {int(u0['friction_pct'])}% feel under-supported).")

    prio_rows = "".join(
        f"<tr><td>{_html.escape(r['key'])}</td><td>{int(r['volume'])}</td>"
        f"<td>{int(r['neg_share']*100)}%</td></tr>"
        for _, r in P.iterrows()) or "<tr><td colspan='3'>Not enough tagged comments to rank.</td></tr>"
    def _c(v):
        return RYG_HEX.get(ryg_of(v), "#333")
    def _status_color(r):
        """Status word + color: importance dims use RYG; agreement-only (Tech) use the band."""
        if pd.isna(r["gap5"]):
            lab, hx = tech_status(r["mean_support"])
            return lab, hx
        return STATUS_WORD.get(ryg_of(r["gap5"]), ""), _c(r["gap5"])
    dsum_rows = ""
    for _, r in d["dim_summary"].iterrows():
        gpv = r["gap5"]; status, col = _status_color(r)
        dsum_rows += (f"<tr><td>{_html.escape(r['dimension'])}</td>"
                      f"<td>{_f1(r['mean_importance'])}</td>"
                      f"<td>{_f1(r['mean_support'])}</td>"
                      f"<td style='color:{col};font-weight:700'>{_gp(gpv)}</td>"
                      f"<td style='color:{col};font-weight:600'>{status}</td></tr>")
    dsum_rows = dsum_rows or "<tr><td colspan='5'>Not one of the 10 in-scope sites.</td></tr>"
    # Associate Sentiment section rows (standalone) — the Q8/Q12/Q16/Q19 summary items
    AREA_SHORT2 = {"Work Activity": "Work Activity", "Physical Feature": "Physical space",
                   "Service Level": "Services", "Digital & Tech": "Technology"}
    sent_rows = ""
    for a in d.get("agreement", []):
        sm = a.get("site_mean"); om = a.get("org_mean")
        if pd.isna(sm):
            continue
        sent_rows += (f"<tr><td>{_html.escape(AREA_SHORT2.get(a['dim'], a['dim']))}</td>"
                      f"<td class='q'>{_html.escape(a.get('q',''))}</td>"
                      f"<td style='font-weight:700'>{sm:.2f}</td>"
                      f"<td>{'' if pd.isna(om) else f'{om:.2f}'}</td>"
                      f"<td style='color:{a.get('hex','#888')};font-weight:600'>{a.get('verdict','')}</td></tr>")
    so = d.get("sentiment_overall")
    if so and pd.notna(so.get("site_mean")):
        pstr = f", p={so['p']:.2f}" if pd.notna(so.get("p")) else ""
        sent_rows += (f"<tr class='ovr'><td><b>Overall</b></td><td class='q'>Average of the four items</td>"
                      f"<td style='font-weight:800'>{so['site_mean']:.2f}</td>"
                      f"<td>{so['org_mean']:.2f}</td>"
                      f"<td style='color:{so['hex']};font-weight:700'>{so['verdict']}{pstr}</td></tr>")
    sent_rows = sent_rows or "<tr><td colspan='5'>Not one of the 10 in-scope sites.</td></tr>"
    if len(d["detail"]):
        detail_rows = ""
        for dim in d["detail"]["dimension"].drop_duplicates():
            detail_rows += f"<tr class='dh'><td colspan='5'>{_html.escape(dim)}</td></tr>"
            for _, r in d["detail"][d["detail"]["dimension"] == dim].iterrows():
                status, col = _status_color(r)
                detail_rows += (f"<tr><td class='tp'>{_html.escape(str(r['item']))}</td>"
                                f"<td style='color:{col}'>{_f1(r['mean_importance'])}</td>"
                                f"<td style='color:{col}'>{_f1(r['mean_support'])}</td>"
                                f"<td style='color:{col};font-weight:700'>{_gp(r['gap5'])}</td>"
                                f"<td style='color:{col};font-weight:600'>{status}</td></tr>")
    else:
        detail_rows = ("<tr><td colspan='5'>Not one of the 10 in-scope sites — no official "
                       "scores. The comment analysis above still applies.</td></tr>")

    # Report-card header — raw composite + relative standing (no absolute letter grade)
    AREA_SHORT = {"Work Activity": "Work Activity", "Physical Feature": "Physical",
                  "Service Level": "Services", "Digital & Tech": "Technology"}
    card_html = ""
    if d.get("card"):
        cc = d["card"]
        oh = cc.get("tier_hex", "#5A9E3A")
        ovm = vs_mark(cc["overall"], cc["org_overall"])
        badges = ""
        for a in cc["areas"]:
            ah = a.get("hex", "#888")
            rank_chip = f"#{a['rank']} of {a['n']}" if a.get("rank") else ""
            badges += (f"<div class='area'><div class='an'>{_html.escape(AREA_SHORT.get(a['dim'], a['dim']))}</div>"
                       f"<div class='ab' style='background:{ah}'>{a['score']:.0f}</div>"
                       f"<div class='as'>{rank_chip}</div>"
                       f"<div class='am'>{vs_mark(a['score'], a['org'])}</div></div>")
        rank_str = (f"Rank <b>{cc['rank']} of {cc['n_sites']}</b> · {cc['tier']} of sites"
                    if cc.get("rank") else "")
        org_str = f"Company average {cc['org_overall']:.0f}/100" if pd.notna(cc.get("org_overall")) else ""
        rng = (f" · sites range {cc['dist_min']:.0f}–{cc['dist_max']:.0f}"
               if pd.notna(cc.get("dist_min")) else "")
        allp = d.get("all_participants") or 0
        part_str = (f"<b>{d['participation']:,}</b> of {allp:,} respondents (this site vs. all 10 sites)"
                    if allp else f"{d['participation']:,} survey participants")
        card_html = (
            f"<div class='rc'><div class='rc-o'>"
            f"<div class='rc-badge' style='background:{oh}'>{cc['overall']:.0f}</div>"
            f"<div><div class='rc-t'>Overall PulsePoint Experience Index Score (0–100) · {cc['tier']} of sites</div>"
            f"<div class='rc-s'>{rank_str} &nbsp; {ovm}</div>"
            f"<div class='rc-p'>{org_str}{rng}</div>"
            f"<div class='rc-p'>{part_str} · {d['n']:,} written comments</div>"
            f"</div></div>"
            f"<div class='rc-dimlab'>Dimension Scores (0–100)</div>"
            f"<div class='rc-areas'>{badges}</div>"
            f"<div class='rc-key'>The <b>Index Score</b> is the average of the four Dimension Scores (0–100) — the "
            f"same calculation as the Tableau deck. Area boxes show each dimension's score and its rank among the 10 "
            f"sites. <b>Tier</b> = thirds of the 10-site distribution (relative standing, not an absolute cutoff). "
            f"&#9650;/&#9660; = vs. the company average.</div></div>")

    # All-comments appendix — by category then sentiment (collapsible)
    appendix_html = ""
    rdf = d.get("real_df")
    if rdf is not None and len(rdf):
        _r = rdf.reset_index(drop=True).copy(); _r["_rid"] = range(len(_r))
        ex = _r.assign(_cat=_r["eff_themes"].str.split("|")).explode("_cat")
        ex["_cat"] = ex["_cat"].str.strip().map(lambda t: THEME_TO_CAT.get(t, "Uncategorized"))
        ex = ex[ex["_cat"] != "Uncategorized"].drop_duplicates(subset=["respondent", "comment", "_cat"])
        cat_order = [c for c in (meta.get("categories") or []) if c in set(ex["_cat"])]
        SENTS = [("Negative", "#C0392B"), ("Neutral / Mixed", "#7A7A7A"), ("Positive", "#2E7D32")]
        for cat in cat_order:
            sub = ex[ex["_cat"] == cat]
            groups = ""
            for sname, scol in SENTS:
                ssub = sub[sub["eff_sentiment"] == sname]
                if not len(ssub):
                    continue
                lis = "".join(
                    f"<li>{_html.escape(str(r['comment']))}"
                    f"<span class='cmeta'> — {_html.escape(str(r['role']))}, {_html.escape(str(r['tenure']))}</span></li>"
                    for _, r in ssub.iterrows())
                groups += (f"<div class='sgrp'><div class='slab' style='color:{scol}'>"
                           f"{sname} · {len(ssub)}</div><ul class='clist'>{lis}</ul></div>")
            appendix_html += (f"<details class='catblk'><summary>{_html.escape(cat)} "
                              f"<span class='cct'>({len(sub)})</span></summary>{groups}</details>")
    fix_rows = "".join(f"<li><b>{_html.escape(t)}</b> — {v} comments, {int(ns*100)}% negative</li>"
                       for t, v, ns in d["fixes"]) or "<li>No clearly negative themes.</li>"
    str_rows = "".join(f"<li><b>{_html.escape(t)}</b> — {v} comments, {int(ps*100)}% positive</li>"
                       for t, v, ps in d["strengths"]) or "<li>No clearly positive themes.</li>"
    bench_rows = "".join(
        f"<tr><td>{_html.escape(t)}</td><td>{sr*100:.0f}%</td><td>{orr*100:.0f}%</td></tr>"
        for t, sr, orr in d["bench"])
    score_rows = "".join(
        f"<tr><td>{_html.escape(lab)}</td><td>{sv:.2f}</td><td>{ov:.2f}</td>"
        f"<td style='color:{'#D1495B' if sv<ov-0.05 else ('#2E9E6B' if sv>ov+0.05 else '#888')}'>"
        f"{sv-ov:+.2f}</td></tr>" for lab, sv, ov in d["scores"])

    def quote_html(q):
        rep = f" · represents ~{q['represents']} similar comments" if q.get("represents", 0) >= 3 else ""
        return (f"<blockquote>{_html.escape(q['text'])}"
                f"<cite>— {_html.escape(q['role'])}, {_html.escape(q['tenure'])} · "
                f"<b>{_html.escape(q['theme'])}</b>{rep}</cite></blockquote>")
    neg_quotes = "".join(quote_html(q) for q in d["quotes"]) or "<p style='color:#888'>No representative critical quotes.</p>"
    pos_quotes = "".join(
        f"<blockquote class='pos'>{_html.escape(q['text'])}"
        f"<cite>— {_html.escape(q['role'])}, {_html.escape(q['tenure'])} · <b>{_html.escape(q['theme'])}</b></cite></blockquote>"
        for q in d["pos_quotes"])
    img = lambda k: (f"<img src='data:image/png;base64,{charts[k]}'/>" if k in charts else "")

    # demographics — visual rollup first, detailed table second (+ explanatory note)
    demo = d.get("demographics") or {}
    demo_total = d.get("demo_total") or 0
    demo_html = ""
    if demo and demo_total:
        cols = ""
        for label, vc in demo.items():
            rows = "".join(
                f"<tr><td>{_html.escape(str(k))}</td><td>{int(v)}</td>"
                f"<td>{v / demo_total * 100:.0f}%</td></tr>" for k, v in vc.items())
            cols += (f"<div><table><tr><th>{_html.escape(label)}</th><th>#</th><th>%</th></tr>{rows}</table></div>")
        demo_html = (f"<p class='lgd'>Demographics are provided to help readers understand the respondent mix behind "
                     f"the results — {demo_total:,} respondents at this site; % is each group's share of the site.</p>"
                     f"<div class='charts'>{img('demo')}</div>"
                     f"<div class='demo'>{cols}</div>")
    # comment-sentiment headline (used in the Open-ended comment themes section)
    cs = (f"<span style='color:{SENT_HEX['Positive']}'>{d['pos']:.0f}% positive</span> · "
          f"<span style='color:#7a7a7a'>{d['neu']:.0f}% neutral</span> · "
          f"<span style='color:{SENT_HEX['Negative']}'>{d['neg']:.0f}% negative</span>")

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>PulsePoint — {_html.escape(site)}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1a1a2e;
   max-width:860px;margin:24px auto;padding:0 22px;line-height:1.5}}
 h1{{font-size:24px;margin-bottom:2px}} .sub{{color:#666;font-size:13px;margin-bottom:14px}}
 h3{{border-bottom:2px solid #eee;padding-bottom:4px;margin-top:24px;font-size:15px;text-transform:uppercase;letter-spacing:.03em;color:#2b3a67}}
 .narr{{background:#f6f8fb;border-left:4px solid #4C78A8;padding:12px 14px;border-radius:4px;font-size:14px}}
 .charts{{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:10px}}
 .charts img{{max-height:200px}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}}
 th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid #eee}} th{{color:#555;font-weight:600}}
 td:nth-child(n+2),th:nth-child(n+2){{text-align:right}}
 .two{{display:flex;gap:26px}} .two>div{{flex:1}}
 ul{{margin:6px 0;padding-left:18px;font-size:13px}} li{{margin:3px 0}}
 blockquote{{margin:7px 0;padding:8px 12px;background:#fafafa;border-left:3px solid #D1495B;font-size:13px;font-style:italic}}
 blockquote.pos{{border-left-color:#2E9E6B}}
 cite{{display:block;font-style:normal;color:#888;font-size:11px;margin-top:4px}}
 tr.dh td{{background:#eef1f7;font-weight:700;color:#2b3a67;text-align:left}}
 table.detail td.tp{{max-width:380px}}
 .lgd{{font-size:11px;color:#888;margin-top:5px}}
 .rc{{border:1px solid #e3e6ec;border-radius:10px;padding:16px 18px;margin:4px 0 16px;background:#fbfcfe}}
 .rc-o{{display:flex;align-items:center;gap:14px}}
 .rc-badge{{width:62px;height:62px;border-radius:10px;color:#fff;font-size:27px;font-weight:800;
   display:flex;align-items:center;justify-content:center;flex-shrink:0}}
 .rc-t{{font-size:15px;font-weight:700;color:#2b3a67}} .rc-s{{font-size:14px;margin-top:1px}}
 .rc-p{{font-size:12px;color:#777;margin-top:3px}}
 .rc-sent{{font-size:13px;font-weight:600;margin:6px 0}}
 .rc-dimlab{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.03em;color:#6b7280;margin-top:14px;margin-bottom:2px}}
 table.sent td.q{{color:#555;font-size:12px;max-width:360px;white-space:normal;text-align:left}}
 table.sent th:nth-child(2){{text-align:left}}
 table.sent tr.ovr td{{border-top:2px solid #cbd5e1;background:#f7f9fc}}
 .rc-areas{{display:flex;gap:10px;margin-top:4px;flex-wrap:wrap}}
 .area{{flex:1;min-width:104px;text-align:center;border:1px solid #eee;border-radius:8px;padding:9px 6px}}
 .an{{font-size:11.5px;color:#555;margin-bottom:6px;height:26px;display:flex;align-items:center;justify-content:center}}
 .ab{{width:40px;height:34px;border-radius:7px;color:#fff;font-weight:800;font-size:15px;
   display:flex;align-items:center;justify-content:center;margin:0 auto}}
 .as{{font-size:13px;margin-top:5px;font-weight:600}} .am{{font-size:11px;margin-top:2px}}
 .rc-key{{font-size:10.5px;color:#999;margin-top:12px;border-top:1px solid #eee;padding-top:8px}}
 details.catblk{{border:1px solid #e6e8ee;border-radius:7px;margin:6px 0;padding:2px 10px;background:#fcfcfd}}
 details.catblk>summary{{cursor:pointer;font-weight:700;color:#2b3a67;padding:7px 0;font-size:13px}}
 .cct{{color:#999;font-weight:400;font-size:12px}}
 .sgrp{{margin:4px 0 8px}} .slab{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.03em;margin:6px 0 2px}}
 ul.clist{{margin:2px 0 6px;padding-left:18px}} ul.clist li{{font-size:12.5px;margin:3px 0;line-height:1.45}}
 .cmeta{{color:#999;font-size:11px}}
 .demo{{display:flex;gap:18px;flex-wrap:wrap;margin-top:8px}} .demo>div{{flex:1;min-width:210px}}
 .demo table{{font-size:12px}}
 .foot{{color:#888;font-size:11px;margin-top:24px;border-top:1px solid #eee;padding-top:8px}}
 @page{{size:Letter;margin:12mm}}
 @media print{{ body{{max-width:100%;margin:0;padding:0}} .charts img{{max-width:100%}}
   h1,h3{{break-after:avoid}} tr,blockquote,.rc,.area,details.catblk,.oap-pri{{break-inside:avoid}} }}
{action_plan.ACTION_PLAN_CSS}
</style></head><body>
<h1>PulsePoint Experience — {_html.escape(site)}</h1>
<div class="sub">Workplace experience report card · completed responses</div>
{card_html}
{action_plan_html}
<h3>Demographics — respondent profile</h3>
{demo_html or '<p style="color:#888">No demographic data.</p>'}
<h3>Importance vs. Support gaps</h3>
<table><tr><th>Dimension</th><th>Importance</th><th>Support</th><th>Gap</th><th>Status</th></tr>{dsum_rows}</table>
<div class="rc-dimlab">Detail — by question</div>
<table class="detail"><tr><th>Dimension topic (question)</th><th>Importance</th><th>Support</th><th>Gap</th><th>Status</th></tr>{detail_rows}</table>
<p class="lgd">Gap = Support − Importance (1–5). Status:
<span style="color:#C0392B">■</span> Needs attention · <span style="color:#2E7D32">■</span> On track ·
<span style="color:#E67E22">■</span> Exceeds needs. <b>Digital &amp; Tech</b> is agreement-only (no Importance/Gap); it's
rated on the 1–5 agreement level — <span style="color:#2E7D32">■</span> Strong (≥4.0) ·
<span style="color:#E0A400">■</span> Moderate (3.5–3.9) · <span style="color:#C0392B">■</span> Needs attention (&lt;3.5).</p>
<h3>Associate Sentiment Score</h3>
<p class="lgd">The survey's summary sentiment items — one per dimension (Q8/Q12/Q16/Q19, 1–5; Q16 = Services, Q19 = Technology
by survey position). “Company” is the other in-scope sites; “vs company” is a Welch t-test (significant at p&lt;0.05).</p>
<table class="sent"><tr><th>Dimension</th><th>Summary sentiment question</th><th>Site</th><th>Company</th><th>vs company</th></tr>{sent_rows}</table>
<h3>Friction &amp; strength themes</h3>
<div class="charts">{img('gap')}{img('strength')}</div>
<h3>Open-ended comment themes</h3>
<div class="rc-sent">Comment sentiment: {cs}</div>
<div class="narr">{narrative}</div>
<div class="charts">{img('donut')}{img('themes')}</div>
<h3>Top priorities to address</h3>
<table><tr><th>Category</th><th>#&nbsp;comments</th><th>%&nbsp;negative</th></tr>{prio_rows}</table>
<div class="two"><div>
<h3>What needs attention</h3><ul>{fix_rows}</ul></div><div>
<h3>What's working</h3><ul>{str_rows}</ul></div></div>
<h3>Vs. org — theme rates</h3>
<table><tr><th>Theme</th><th>site</th><th>org</th></tr>{bench_rows}</table>
<p class="lgd"><b>How comments are categorized:</b> themes are assigned by a transparent keyword taxonomy built for this
survey (28 themes across 7 categories), kept separate from the PXIP category set by design so the open-ended read stands
on its own. The mapping lives in <code>taxonomy.py</code> and can be tuned without touching the scores.</p>
<h3>In their words — what to fix</h3>{neg_quotes}
{('<h3>In their words — what’s working</h3>'+pos_quotes) if pos_quotes else ''}
<h3>All comments — by category &amp; sentiment</h3>
<p class="lgd">Every open-ended comment at this site, grouped by category then sentiment. Click a category to expand.
A comment that touches more than one category appears under each.</p>
{appendix_html or '<p style="color:#888">No comments to list.</p>'}
<div class="foot"><b>Data source &amp; method.</b> Built from the PulsePoint survey export ·
{'completed responses only' if completers_only else 'all responses'} · 10 in-scope sites. Support responses map to a
0–100 scale; each dimension is the average of its items; the <b>Overall PulsePoint Experience Index Score</b> is the
<b>unweighted</b> average of the four Dimension Scores — the same calculation as the Tableau deck. Friction &amp; RYG per
the official Tableau spec · Priority = 40% volume + 35% share negative + 25% Friction.</div>
</body></html>"""

@st.cache_data(show_spinner=False)
def build_site_pdf(site, completers_flag):
    """PDF export = an exact render of the HTML site report (identical content and layout)."""
    return html_to_pdf(build_site_report(site))

def build_site_pptx(sites, completers_flag):
    """One slide per site."""
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    prs = Presentation()
    prs.slide_width = Inches(13.33); prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    for site in sites:
        d = site_report_data(site, completers_flag)
        s = prs.slides.add_slide(blank)
        tb = s.shapes.add_textbox(Inches(0.4), Inches(0.25), Inches(12.5), Inches(0.8)).text_frame
        tb.text = f"PulsePoint — {site}"
        tb.paragraphs[0].font.size = Pt(26); tb.paragraphs[0].font.bold = True
        if d["n"] == 0:
            continue
        sub = s.shapes.add_textbox(Inches(0.4), Inches(0.95), Inches(12.5), Inches(0.4)).text_frame
        sub.text = (f"{d['n']} comments · {d['neg']:.0f}% negative, {d['pos']:.0f}% positive · "
                    f"top area: {d['P']['key'].tolist()[0] if len(d['P']) else '—'}")
        sub.paragraphs[0].font.size = Pt(12); sub.paragraphs[0].font.color.rgb = RGBColor(0x66, 0x66, 0x66)
        charts = d["charts"]
        x = Inches(0.4)
        for k in ("donut", "themes", "gap"):
            if k in charts:
                s.shapes.add_picture(io.BytesIO(base64.b64decode(charts[k])), x, Inches(1.5), height=Inches(2.1))
                x += Inches(4.3)
        # priorities table
        rows = min(6, len(d["P"]) + 1)
        tbl = s.shapes.add_table(rows, 4, Inches(0.4), Inches(4.0), Inches(6.0), Inches(2.6)).table
        for j, h in enumerate(["Category", "# comments", "% neg", "priority"]):
            c = tbl.cell(0, j); c.text = h; c.text_frame.paragraphs[0].font.size = Pt(11); c.text_frame.paragraphs[0].font.bold = True
        for i, (_, r) in enumerate(d["P"].head(5).iterrows(), 1):
            for j, v in enumerate([r["key"], int(r["volume"]), f"{int(r['neg_share']*100)}%", int(round(r["Priority"]))]):
                c = tbl.cell(i, j); c.text = str(v); c.text_frame.paragraphs[0].font.size = Pt(10)
        # top quotes
        qtf = s.shapes.add_textbox(Inches(6.7), Inches(4.0), Inches(6.2), Inches(3.0)).text_frame
        qtf.word_wrap = True
        qtf.text = "In their words:"
        qtf.paragraphs[0].font.bold = True; qtf.paragraphs[0].font.size = Pt(12)
        for q in d["quotes"][:3]:
            p = qtf.add_paragraph(); p.text = f"“{q['text'][:150]}” — {q['role']} · {q['theme']}"
            p.font.size = Pt(9); p.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    buf = io.BytesIO(); prs.save(buf)
    return buf.getvalue()

# ===========================================================================
# PORTFOLIO — "10 PulsePoints Report Card" (roll-up across the in-scope sites)
# ===========================================================================
def _portfolio_heatmap(matrix, rows, cols, vmin, vmax, cmap, fmt, title, figsize, xrot=0,
                       cbar_label="", cbar_ticks=None):
    arr = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(arr, aspect="auto", cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=8, rotation=xrot, ha=("right" if xrot else "center"))
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7, color="#1a1a1a")
    ax.set_title(title, fontsize=10.5, weight="bold", color="#2b3a67", loc="left", pad=8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # color key
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    cbar.ax.tick_params(labelsize=6.5, length=0)
    cbar.set_label(cbar_label, fontsize=7.5, color="#555")
    cbar.outline.set_visible(False)
    fig.tight_layout()
    return _b64png(fig)

def _portfolio_charts(dd):
    charts = {}
    board = dd["board"]; sites = [b["site"] for b in board]
    # 1) Leaderboard bar (Index Score, colored by tier; best at top)
    if board:
        order = board[::-1]  # worst first -> best ends at top
        fig, ax = plt.subplots(figsize=(7.4, 3.7))
        ax.barh(range(len(order)), [b["index"] for b in order], color=[b["hex"] for b in order], height=0.66)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([b["site"] for b in order], fontsize=8)
        mx = max(b["index"] for b in board)
        for i, b in enumerate(order):
            ax.text(b["index"] + mx * 0.01, i, f"{b['index']:.0f}  (#{b['rank']})",
                    va="center", fontsize=8, weight="bold", color="#333")
        cx = dd.get("comp_index")
        if pd.notna(cx):
            ax.axvline(cx, color="#555", ls="--", lw=1)
            ax.text(cx, len(order) - 0.4, f" company {cx:.0f}", fontsize=7.5, color="#555")
        ax.set_xlim(0, mx * 1.2); ax.set_xlabel("Index Score (0–100)", fontsize=8.5)
        ax.set_title("Site leaderboard — Overall PulsePoint Experience Index Score", fontsize=11,
                     weight="bold", color="#2b3a67", loc="left")
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        fig.tight_layout(); charts["leaderboard"] = _b64png(fig)
    # 2) Site × dimension heatmap (absolute 0–100, RdYlGn)
    dims = ["Work Activity", "Physical Feature", "Service Level", "Digital & Tech"]
    dbs = dd["dim_by_site"]
    mat = [[dbs.get(dm, {}).get(s, np.nan) for dm in dims] for s in sites]
    if sites:
        charts["heatmap_dim"] = _portfolio_heatmap(
            mat, sites, ["Work", "Physical", "Service", "Tech"], 50, 85, "RdYlGn", "{:.0f}",
            "Dimension Scores by site (0–100; green = higher)", (6.8, 3.9),
            cbar_label="Score (0–100)", cbar_ticks=[50, 60, 70, 80])
    # 3) Site × theme heatmap (% of each site's comments; Blues)
    themes = dd["site_theme_themes"]; sxt = dd["site_theme"]
    if sites and themes:
        tmat = [[sxt.get(s, {}).get(t, np.nan) for t in themes] for s in sites]
        vmax = max([v for row in tmat for v in row if not np.isnan(v)] + [1])
        charts["heatmap_theme"] = _portfolio_heatmap(
            tmat, sites, [_wrap(t, 12) for t in themes], 0, vmax, "Blues", "{:.0f}",
            "Comment theme prevalence by site (% of site's comments)", (8.6, 4.0), xrot=30,
            cbar_label="% of site's comments")
    # 4) Company friction + strengths (reuse the site chart renderer)
    if dd["friction"]:
        fr = pd.DataFrame(dd["friction"])
        fr["friction"] = pd.to_numeric(fr["friction"], errors="coerce")
        fr["friction_pct"] = pd.to_numeric(fr["friction_pct"], errors="coerce")
        charts["friction"] = _score_bar(fr.sort_values("friction"), "friction", "#D1495B",
            "Company-wide highest-Friction items", "Friction (0–100, higher = worse)", "#7a1f2b",
            pctcol="friction_pct", xmax=100)
    if dd["strength"]:
        stg = pd.DataFrame(dd["strength"])
        stg["support_100"] = pd.to_numeric(stg["support_100"], errors="coerce")
        charts["strength"] = _score_bar(stg.sort_values("support_100"), "support_100", "#2E7D32",
            "Company-wide strengths — best-supported items", "Support (0–100, higher = better)", "#1c5024", xmax=100)
    # 5) Company demographics visual
    demo = dd["demographics"]; dt = dd.get("n_participants") or 0
    if demo and dt:
        keys = list(demo.keys())
        fig, axes = plt.subplots(1, len(keys), figsize=(3.05 * len(keys), 2.6))
        if len(keys) == 1:
            axes = [axes]
        for ax, lab in zip(axes, keys):
            vc = demo[lab].head(6)[::-1]; pct = vc / dt * 100
            ax.barh(range(len(vc)), pct.values, color="#5B7DB1", height=0.62)
            ax.set_yticks(range(len(vc))); ax.set_yticklabels([_wrap(str(k), 16) for k in vc.index], fontsize=7)
            for i, v in enumerate(pct.values):
                ax.text(v + 1.5, i, f"{v:.0f}%", va="center", fontsize=7, color="#33475b")
            ax.set_xlim(0, 112); ax.set_title(lab, fontsize=9.5, weight="bold", color="#2b3a67")
            ax.tick_params(axis="x", labelsize=7)
            for s in ["top", "right"]: ax.spines[s].set_visible(False)
        fig.tight_layout(); charts["demo"] = _b64png(fig)
    return charts

@st.cache_data(show_spinner=False)
def portfolio_data(completers_flag):
    """Roll-up data for the 10-site portfolio report card."""
    ten = meta.get("ten_sites", [])
    dd = {}
    sd = scores_dim if scores_dim is not None else pd.DataFrame()
    def _dim_val(name):
        if not sd.empty and (sd["dimension"] == name).any():
            return float(pd.to_numeric(sd.loc[sd["dimension"] == name, "composite"], errors="coerce").iloc[0])
        return np.nan
    dd["comp_index"] = _dim_val("OVERALL (equal)")
    dims = ["Work Activity", "Physical Feature", "Service Level", "Digital & Tech"]
    dd["dim_comp"] = {dm: _dim_val(dm) for dm in dims}
    # per-site index + rank + tier
    stand = site_standings(completers_flag)
    overalls = stand["overall"]; ovals = list(overalls.values())
    board = []
    for site, idx in sorted(overalls.items(), key=lambda kv: -kv[1]):
        r, n, tier, hexc, _p = rank_tier(idx, ovals)
        board.append({"site": site, "index": idx, "rank": r, "n": n, "tier": tier, "hex": hexc})
    dd["board"] = board
    dd["dim_by_site"] = stand["dims"]
    # company dimension gaps (Importance vs Support, 1–5) from item-level scores
    sc = scores if scores is not None else pd.DataFrame()
    gaps = []
    if not sc.empty:
        for dm in dims:
            sub = sc[sc["dimension"] == dm]
            if not len(sub):
                continue
            mi = pd.to_numeric(sub["mean_importance"], errors="coerce").mean()
            ms = pd.to_numeric(sub["mean_support"], errors="coerce").mean()
            gaps.append({"dim": dm, "imp": mi, "sup": ms,
                         "gap": (ms - mi) if pd.notna(mi) else np.nan,
                         "support100": dd["dim_comp"].get(dm, np.nan)})
    dd["dim_gaps"] = gaps
    # per-site associate sentiment (composite 1–5) + company
    rs = resp_scores[resp_scores["site"].isin(ten)] if not resp_scores.empty else pd.DataFrame()
    dd["comp_sentiment"] = float(rs["composite"].mean()) if len(rs) else np.nan
    dd["sent_by_site"] = {s: float(g["composite"].mean()) for s, g in rs.groupby("site")} if len(rs) else {}
    dd["n_participants"] = int(len(rs)); dd["n_sites"] = len(board)
    # company friction items + strengths
    if not sc.empty:
        imp_items = sc[pd.to_numeric(sc["mean_importance"], errors="coerce").notna()].copy()
        imp_items["friction"] = pd.to_numeric(imp_items["friction"], errors="coerce")
        dd["friction"] = imp_items.sort_values("friction", ascending=False).head(6)[
            ["dimension", "item", "friction", "friction_pct"]].to_dict("records")
        sc2 = sc.copy(); sc2["support_100"] = pd.to_numeric(sc2["support_100"], errors="coerce")
        dd["strength"] = sc2.sort_values("support_100", ascending=False).head(6)[
            ["dimension", "item", "support_100"]].to_dict("records")
    else:
        dd["friction"] = []; dd["strength"] = []
    # open-ended: company comment sentiment + top themes + site×theme
    real = comments[~comments["eff_junk"]]
    real10 = real[real["site"].isin(ten)]
    dd["n_comments"] = int(len(real10)); dd["n_commenters"] = int(real10["respondent"].nunique())
    dd["pos"] = float((real10["eff_sentiment"] == "Positive").mean() * 100) if len(real10) else 0.0
    dd["neg"] = float((real10["eff_sentiment"] == "Negative").mean() * 100) if len(real10) else 0.0
    dd["neu"] = 100 - dd["pos"] - dd["neg"]
    tex = explode_level(real10, "Themes")
    tcounts = tex["key"].value_counts()
    tops = list(tcounts.head(8).index)
    dd["top_themes"] = [{"theme": t, "n": int(tcounts[t]), "pct": tcounts[t] / max(1, len(real10)) * 100,
                         "neg": float((tex[tex["key"] == t]["eff_sentiment"] == "Negative").mean() * 100)}
                        for t in tops]
    site_tot = real10.groupby("site")["comment_id"].nunique()
    dd["site_theme_themes"] = tops
    dd["site_theme"] = {b["site"]: {t: (tex[(tex["site"] == b["site"]) & (tex["key"] == t)]["comment_id"].nunique()
                                        / max(1, int(site_tot.get(b["site"], 0))) * 100) for t in tops}
                        for b in board}
    # company demographics
    demo = {}
    for lab, col in [("Role", "role"), ("Tenure", "tenure"), ("Hybrid category", "hybrid")]:
        if len(rs) and col in rs.columns:
            vc = rs[col].dropna().astype(str).value_counts()
            if len(vc):
                demo[lab] = vc
    dd["demographics"] = demo
    dd["charts"] = _portfolio_charts(dd)
    return dd

def _sent_mark(v, ref, thr=0.05):
    if pd.isna(v) or pd.isna(ref):
        return ""
    d = v - ref
    if d >= thr:
        return f"<span style='color:#1E7D32'>&#9650; +{d:.2f}</span>"
    if d <= -thr:
        return f"<span style='color:#C0392B'>&#9660; {d:.2f}</span>"
    return "<span style='color:#888'>&#8776;</span>"

def build_portfolio_report(completers_flag):
    """Self-contained HTML — the 10 PulsePoints portfolio report card (exec brief)."""
    dd = portfolio_data(completers_flag)
    ch = dd["charts"]
    img = lambda k: (f"<img src='data:image/png;base64,{ch[k]}'/>" if k in ch else "")
    cx = dd["comp_index"]; board = dd["board"]
    ivals = [b["index"] for b in board]
    rng = f"{min(ivals):.0f}–{max(ivals):.0f}" if ivals else "—"
    # leaderboard table
    lead = ""
    for b in board:
        s = dd["sent_by_site"].get(b["site"], np.nan)
        lead += (f"<tr><td>{_html.escape(b['site'])}</td>"
                 f"<td style='font-weight:800'>{b['index']:.0f}</td>"
                 f"<td>#{b['rank']} of {b['n']}</td>"
                 f"<td style='color:{b['hex']};font-weight:600'>{b['tier']}</td>"
                 f"<td>{vs_mark(b['index'], cx)}</td>"
                 f"<td>{'—' if pd.isna(s) else f'{s:.2f}'} {_sent_mark(s, dd['comp_sentiment'])}</td></tr>")
    # dimension gaps table
    def _dstat(g):
        if pd.isna(g["gap"]):
            lab, hx = tech_status(g["sup"]); return lab, hx
        ry = ryg_of(g["gap"]); return STATUS_WORD.get(ry, ""), RYG_HEX.get(ry, "#333")
    gap_rows = ""
    for g in dd["dim_gaps"]:
        lab, hx = _dstat(g)
        gap_rows += (f"<tr><td>{_html.escape(g['dim'])}</td><td>{_f1(g['imp'])}</td><td>{_f1(g['sup'])}</td>"
                     f"<td style='color:{hx};font-weight:700'>{_gp(g['gap'])}</td>"
                     f"<td style='color:{hx};font-weight:600'>{lab}</td>"
                     f"<td style='font-weight:700'>{g['support100']:.0f}</td></tr>")
    # top themes table
    th_rows = "".join(
        f"<tr><td>{_html.escape(t['theme'])}</td><td>{t['pct']:.0f}%</td><td>{t['n']:,}</td>"
        f"<td style='color:{'#C0392B' if t['neg']>=50 else '#555'}'>{t['neg']:.0f}% neg</td></tr>"
        for t in dd["top_themes"])
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>10 PulsePoints — Portfolio Report Card</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1a1a2e;max-width:900px;
   margin:22px auto;padding:0 22px;line-height:1.5}}
 h1{{font-size:23px;margin-bottom:2px}} .sub{{color:#666;font-size:13px;margin-bottom:14px}}
 h3{{border-bottom:2px solid #eee;padding-bottom:4px;margin-top:22px;font-size:14px;text-transform:uppercase;
   letter-spacing:.03em;color:#2b3a67}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}}
 th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid #eee}} th{{color:#555;font-weight:600}}
 td:nth-child(n+2),th:nth-child(n+2){{text-align:right}}
 .charts img{{max-width:100%;margin-top:8px}}
 .rc{{border:1px solid #e3e6ec;border-radius:10px;padding:16px 18px;margin:4px 0 14px;background:#fbfcfe;display:flex;
   align-items:center;gap:16px}}
 .rc-badge{{width:66px;height:66px;border-radius:10px;background:#2b3a67;color:#fff;font-size:28px;font-weight:800;
   display:flex;align-items:center;justify-content:center;flex-shrink:0}}
 .rc-t{{font-size:15px;font-weight:700;color:#2b3a67}} .rc-p{{font-size:12.5px;color:#777;margin-top:3px}}
 .lgd{{font-size:11px;color:#888;margin-top:5px}}
 .foot{{color:#888;font-size:11px;margin-top:22px;border-top:1px solid #eee;padding-top:8px}}
 @page{{size:Letter;margin:12mm}}
 @media print{{ body{{max-width:100%;margin:0;padding:0}} .charts img{{max-width:100%}}
   h1,h3{{break-after:avoid}} tr,.rc{{break-inside:avoid}} }}
</style></head><body>
<h1>10 PulsePoints — Portfolio Report Card</h1>
<div class="sub">Company-wide exec brief across the {dd['n_sites']} in-scope sites · completed responses</div>
<div class="rc"><div class="rc-badge">{cx:.0f}</div><div>
 <div class="rc-t">Company Overall PulsePoint Experience Index Score (0–100)</div>
 <div class="rc-p">Sites range {rng} · {dd['n_participants']:,} respondents · {dd['n_comments']:,} substantive comments
  from {dd['n_commenters']:,} associates</div>
 <div class="rc-p">Company comment sentiment:
  <span style="color:#2E9E6B">{dd['pos']:.0f}% positive</span> ·
  <span style="color:#7a7a7a">{dd['neu']:.0f}% neutral</span> ·
  <span style="color:#D1495B">{dd['neg']:.0f}% negative</span></div>
</div></div>
<h3>Site leaderboard</h3>
<table><tr><th>Site</th><th>Index</th><th>Rank</th><th>Tier</th><th>vs company</th><th>Sentiment (1–5)</th></tr>{lead}</table>
<div class="charts">{img('leaderboard')}</div>
<h3>Dimension Scores by site</h3>
<div class="charts">{img('heatmap_dim')}</div>
<p class="lgd">Each cell is that site's Dimension Score (0–100). Color is absolute (green = higher, red = lower);
Work Activity is the company's weakest dimension, so it reads amber/red across most sites.</p>
<h3>Company dimensions — Importance vs. Support</h3>
<table><tr><th>Dimension</th><th>Importance</th><th>Support</th><th>Gap</th><th>Status</th><th>Score (0–100)</th></tr>{gap_rows}</table>
<p class="lgd">Digital &amp; Tech is agreement-only (Strong ≥4.0 / Moderate 3.5–3.9 / Needs attention &lt;3.5).</p>
<h3>Friction &amp; strength themes (company-wide)</h3>
<div class="charts">{img('friction')}{img('strength')}</div>
<h3>Open-ended comment themes</h3>
<table><tr><th>Theme</th><th>% of comments</th><th>Mentions</th><th>Sentiment</th></tr>{th_rows}</table>
<div class="charts">{img('heatmap_theme')}</div>
<p class="lgd">Prevalence = share of each site's comments mentioning the theme (a comment can touch several).
Themes come from a keyword taxonomy (29 themes across 7 categories), kept separate from PXIP categories by design.</p>
<h3>Demographics — respondent profile</h3>
<p class="lgd">Provided to help readers understand the respondent mix behind the results ({dd['n_participants']:,} respondents).</p>
<div class="charts">{img('demo')}</div>
<div class="foot"><b>Data source &amp; method.</b> PulsePoint survey export ·
{'completed responses only' if completers_flag else 'all responses'} · {dd['n_sites']} in-scope sites. The Index Score is
the unweighted average of the four Dimension Scores (0–100) — the same calculation as the Tableau deck. Friction &amp; RYG
per the official Tableau spec.</div>
</body></html>"""

@st.cache_data(show_spinner=False)
def build_portfolio_pdf(completers_flag):
    """PDF export = an exact render of the HTML portfolio report (identical content and layout)."""
    return html_to_pdf(build_portfolio_report(completers_flag))

def _safe_pdf_button(container, label, build_fn, filename, key=None):
    """PDF download button that degrades gracefully: if no PDF renderer is installed,
    show a one-line hint instead of throwing and taking down the whole tab."""
    try:
        data = build_fn()
    except Exception:
        container.caption("PDF export needs a renderer — run `playwright install chromium` once. "
                          "In the meantime, download the HTML and use your browser's Print → Save as PDF "
                          "(identical output).")
        return
    container.download_button(label, data, filename, "application/pdf",
                              use_container_width=True, key=key)

def verbatim_export(site=None):
    """Long verbatim table grouped by theme for CSV/Excel export."""
    base = comments[~comments["eff_junk"]]
    if site:
        base = base[base["site"] == site]
    rows = []
    for _, r in base.iterrows():
        for t in str(r["eff_themes"]).split("|"):
            t = t.strip()
            if t and t != "Unclassified":
                rows.append({"theme": t, "category": THEME_TO_CAT.get(t, "Uncategorized"),
                             "site": r["site"], "role": r["role"], "tenure": r["tenure"],
                             "sentiment": r["eff_sentiment"], "question": r["source_question"],
                             "comment": r["comment"]})
    return pd.DataFrame(rows).sort_values(["theme", "sentiment"])

@st.cache_data(show_spinner=False)
def build_excel_workbook(completers_flag):
    """Multi-sheet analysis workbook."""
    real = comments[~comments["eff_junk"]]
    tex = explode_level(real, "Themes")
    # theme summary
    tsum = tex.groupby("key").agg(
        mentions=("comment_id", "nunique"),
        pct_negative=("eff_sentiment", lambda s: round((s == "Negative").mean() * 100, 1)),
        pct_positive=("eff_sentiment", lambda s: round((s == "Positive").mean() * 100, 1))).reset_index()
    tsum["category"] = tsum["key"].map(lambda t: THEME_TO_CAT.get(t, "Uncategorized"))
    tsum = tsum.rename(columns={"key": "theme"}).sort_values("mentions", ascending=False)
    site_theme = pd.crosstab(tex["site"], tex["key"])
    sent_theme = pd.crosstab(tex["key"], tex["eff_sentiment"])
    comments_out = real[["site", "role", "tenure", "source_question",
                         "eff_sentiment", "eff_themes", "comment"]].rename(
        columns={"eff_sentiment": "sentiment", "eff_themes": "themes"})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        comments_out.to_excel(w, sheet_name="Comments", index=False)
        tsum.to_excel(w, sheet_name="Theme summary", index=False)
        site_theme.to_excel(w, sheet_name="Site x Theme")
        sent_theme.to_excel(w, sheet_name="Sentiment x Theme")
        verbatim_export().to_excel(w, sheet_name="Verbatims by theme", index=False)
        if not scores.empty:
            scores.to_excel(w, sheet_name="Scores & Friction", index=False)
            scores_dim.to_excel(w, sheet_name="Dimension scores", index=False)
    return buf.getvalue()

@st.cache_data(show_spinner=False)
def build_standalone_dashboard(completers_flag):
    """Self-contained HTML dashboard (embedded data + vanilla JS filtering)."""
    real = comments[~comments["eff_junk"]].copy()
    recs = [{"s": r["site"], "r": r["role"], "t": r["tenure"], "q": r["source_question"],
             "m": r["eff_sentiment"], "h": [x.strip() for x in str(r["eff_themes"]).split("|") if x.strip()],
             "c": str(r["comment"])} for _, r in real.iterrows()]
    payload = json.dumps(recs, ensure_ascii=False)
    sites = sorted(real["site"].unique().tolist())
    roles = sorted(real["role"].dropna().unique().tolist())
    tenures = sorted(real["tenure"].dropna().unique().tolist())
    themes = sorted({t for rec in recs for t in rec["h"]})
    opts = lambda xs: "".join(f"<option>{_html.escape(str(x))}</option>" for x in xs)
    scope = "completed responses only" if completers_flag else "all responses"
    return """<!doctype html><html><head><meta charset="utf-8"><title>PulsePoint Comment Browser</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;color:#1a1a2e;background:#f7f8fa}
 header{background:#2b3a67;color:#fff;padding:14px 22px} header h1{margin:0;font-size:19px}
 header .sub{font-size:12px;opacity:.8}
 .wrap{display:flex;gap:16px;padding:16px 22px}
 .filters{width:240px;flex-shrink:0} .filters label{display:block;font-size:11px;color:#666;margin:10px 0 3px;text-transform:uppercase;letter-spacing:.03em}
 select,input{width:100%;padding:6px;border:1px solid #ccc;border-radius:5px;font-size:13px;box-sizing:border-box}
 .main{flex:1;min-width:0}
 .stat{display:inline-block;background:#fff;border:1px solid #e6e6e6;border-radius:8px;padding:8px 14px;margin:0 8px 8px 0}
 .stat b{font-size:20px;display:block}
 .bars{background:#fff;border:1px solid #e6e6e6;border-radius:8px;padding:12px;margin-bottom:12px}
 .bar{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px}
 .bar .lab{width:180px;text-align:right;color:#333} .bar .track{flex:1;background:#eef;border-radius:3px;height:14px}
 .bar .fill{background:#4C78A8;height:14px;border-radius:3px}
 .cmt{background:#fff;border:1px solid #e9e9e9;border-left:4px solid #9AA0A6;border-radius:6px;padding:9px 12px;margin:7px 0;font-size:13px}
 .cmt.Negative{border-left-color:#D1495B} .cmt.Positive{border-left-color:#2E9E6B}
 .cmt .meta{font-size:11px;color:#888;margin-top:5px}
 .chip{display:inline-block;background:#eef1f7;color:#33406b;border-radius:10px;padding:1px 8px;font-size:10px;margin-right:4px}
 .pill{font-weight:600;font-size:11px;padding:1px 7px;border-radius:10px;color:#fff}
 .pill.Negative{background:#D1495B}.pill.Positive{background:#2E9E6B}.pill{background:#9AA0A6}
</style></head><body>
<header><h1>PulsePoint — Comment Browser</h1><div class="sub">Self-contained · """ + scope + """ · filter, search, and read every comment offline</div></header>
<div class="wrap">
 <div class="filters">
  <label>Search text</label><input id="kw" placeholder="keyword…" oninput="render()">
  <label>Site</label><select id="fs" onchange="render()"><option value="">All</option>""" + opts(sites) + """</select>
  <label>Role</label><select id="fr" onchange="render()"><option value="">All</option>""" + opts(roles) + """</select>
  <label>Tenure</label><select id="ft" onchange="render()"><option value="">All</option>""" + opts(tenures) + """</select>
  <label>Sentiment</label><select id="fm" onchange="render()"><option value="">All</option><option>Positive</option><option>Neutral / Mixed</option><option>Negative</option></select>
  <label>Theme</label><select id="fh" onchange="render()"><option value="">All</option>""" + opts(themes) + """</select>
 </div>
 <div class="main">
  <div id="stats"></div>
  <div class="bars" id="themebars"></div>
  <div id="list"></div>
 </div>
</div>
<script>
const DATA = """ + payload + """;
function render(){
 const kw=document.getElementById('kw').value.toLowerCase();
 const fs=document.getElementById('fs').value, fr=document.getElementById('fr').value,
  ft=document.getElementById('ft').value, fm=document.getElementById('fm').value, fh=document.getElementById('fh').value;
 let rows=DATA.filter(d=>(!fs||d.s===fs)&&(!fr||d.r===fr)&&(!ft||d.t===ft)&&(!fm||d.m===fm)&&(!fh||d.h.includes(fh))&&(!kw||d.c.toLowerCase().includes(kw)));
 const n=rows.length, neg=rows.filter(d=>d.m==='Negative').length, pos=rows.filter(d=>d.m==='Positive').length;
 document.getElementById('stats').innerHTML=
  `<div class="stat"><b>${n.toLocaleString()}</b>comments</div>`+
  `<div class="stat"><b style="color:#D1495B">${n?Math.round(neg/n*100):0}%</b>negative</div>`+
  `<div class="stat"><b style="color:#2E9E6B">${n?Math.round(pos/n*100):0}%</b>positive</div>`;
 const tc={};rows.forEach(d=>d.h.forEach(h=>tc[h]=(tc[h]||0)+1));
 const top=Object.entries(tc).sort((a,b)=>b[1]-a[1]).slice(0,12);
 const mx=top.length?top[0][1]:1;
 document.getElementById('themebars').innerHTML='<b style="font-size:12px">Top themes in this view</b>'+
  top.map(([t,c])=>`<div class="bar"><span class="lab">${t}</span><span class="track"><span class="fill" style="width:${c/mx*100}%"></span></span><span>${c}</span></div>`).join('');
 const show=rows.slice(0,400);
 document.getElementById('list').innerHTML=show.map(d=>
  `<div class="cmt ${d.m}"><span class="pill ${d.m}">${d.m}</span> ${esc(d.c)}<div class="meta">${d.h.map(h=>`<span class="chip">${h}</span>`).join('')} · ${d.s} · ${d.r} · ${d.t}</div></div>`).join('')
  +(rows.length>400?`<p style="color:#888">Showing first 400 of ${rows.length.toLocaleString()} — narrow the filters to see more.</p>`:'');
}
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
render();
</script></body></html>"""

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.title("🔎 Filters")

# --- Completion filter (default: completed responses only) ------------------
n_comp = meta.get("n_completers", meta["n_responses"])
completers_only = st.sidebar.toggle(
    "Completed responses only", value=True,
    help=f"Only analyze the {n_comp:,} respondents who reached the end of the survey "
         f"(of {meta['n_responses']:,} total). Turn off to include partial responses.")
if completers_only:
    comments = comments_full[comments_full["reached_end"] == 1].copy()
    resp_scores = resp_scores_full[resp_scores_full["reached_end"] == 1].copy()
    gap, gap_site = GAPS["comp"], GAPS["comp_site"]
    scores, scores_site, scores_dim = SCORES["comp"], SCORES["comp_site"], SCORES["comp_dim"]
    st.sidebar.caption(f"✓ Completed only · {n_comp:,} respondents")
else:
    comments = comments_full.copy()
    resp_scores = resp_scores_full.copy()
    gap, gap_site = GAPS["all"], GAPS["all_site"]
    scores, scores_site, scores_dim = SCORES["all"], SCORES["all_site"], SCORES["all_dim"]
    st.sidebar.caption(f"Including all {meta['n_responses']:,} respondents")
THEME_FRICTION_ALL = _theme_friction_from(scores)

st.sidebar.divider()
st.sidebar.caption("Filters below apply to Overview, Priorities & Explorer tabs")

def ms(label, options, key):
    return st.sidebar.multiselect(label, options, default=[], key=key,
                                  placeholder="All")

f_site   = ms("Site (PulsePoint)", meta["sites"], "f_site")
f_role   = ms("Role", meta["roles"], "f_role")
f_hybrid = ms("Hybrid category", meta["hybrids"], "f_hybrid")
f_tenure = ms("Tenure", meta["tenures"], "f_tenure")
f_ques   = ms("Survey question", meta["questions"], "f_ques")
f_sent   = ms("Sentiment", ["Positive","Neutral / Mixed","Negative"], "f_sent")
f_theme  = ms("Theme", meta["themes"], "f_theme")
kw = st.sidebar.text_input("Keyword search (in comment)", "")
inc_junk = st.sidebar.checkbox("Include junk / no-comment", value=False)

def apply_filters(df):
    d = df.copy()
    if not inc_junk:
        d = d[~d["eff_junk"]]
    if f_site:   d = d[d["site"].isin(f_site)]
    if f_role:   d = d[d["role"].isin(f_role)]
    if f_hybrid: d = d[d["hybrid"].isin(f_hybrid)]
    if f_tenure: d = d[d["tenure"].isin(f_tenure)]
    if f_ques:   d = d[d["source_question"].isin(f_ques)]
    if f_sent:   d = d[d["eff_sentiment"].isin(f_sent)]
    if f_theme:
        pat = "|".join([t.replace("(","\\(").replace(")","\\)").replace("/","\\/") for t in f_theme])
        d = d[d["eff_themes"].str.contains(pat, case=False, na=False, regex=True)]
    if kw.strip():
        d = d[d["comment"].str.contains(kw.strip(), case=False, na=False, regex=False)]
    return d

fdf = apply_filters(comments)

def explode_themes(df):
    t = df.assign(theme=df["eff_themes"].str.split("|")).explode("theme")
    t["theme"] = t["theme"].str.strip()
    t = t[t["theme"] != ""]
    t["category"] = t["theme"].map(lambda x: THEME_TO_CAT.get(x, "Uncategorized"))
    return t

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("PulsePoint — Open-Ended Response Analyzer")
_nresp = comments["respondent"].nunique()
_nreal = int((~comments["eff_junk"]).sum())
st.caption(f"{_nresp:,} commenters · {_nreal:,} comments with content · {comments['site'].nunique()} sites"
           + ("  ·  completed responses only" if completers_only else "  ·  all responses"))

(tab_over, tab_prio, tab_gap, tab_expl, tab_browse, tab_diag,
 tab_report, tab_tag) = st.tabs(
    ["📊 Overview", "🎯 Action Priorities", "⚖️ Scores & Friction",
     "🏷️ Theme & Sentiment Explorer", "🔎 Browse & Pivot", "🔥 Hotspots & Drivers",
     "📄 Site Reports", "✏️ Tagging Workspace"])

# ===========================================================================
# TAB 1 — OVERVIEW
# ===========================================================================
with tab_over:
    st.subheader("Overview")
    n_real = int((~fdf["eff_junk"]).sum()) if inc_junk else len(fdf)
    view = fdf[~fdf["eff_junk"]] if not inc_junk else fdf
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Comments (filtered)", f"{len(view):,}")
    pos = (view["eff_sentiment"]=="Positive").mean()*100 if len(view) else 0
    neg = (view["eff_sentiment"]=="Negative").mean()*100 if len(view) else 0
    c2.metric("Positive", f"{pos:.0f}%")
    c3.metric("Negative", f"{neg:.0f}%")
    c4.metric("Sites represented", view["site"].nunique())
    reviewed = int(comments["reviewed"].astype(str).isin(["True","true","1"]).sum())
    c5.metric("Reviewed", f"{reviewed:,}")

    st.divider()
    a, b = st.columns(2)
    with a:
        st.markdown("**Sentiment mix**")
        sc = view["eff_sentiment"].value_counts().reset_index()
        sc.columns = ["sentiment","n"]
        chart = alt.Chart(sc).mark_arc(innerRadius=55).encode(
            theta="n:Q",
            color=alt.Color("sentiment:N",
                scale=alt.Scale(domain=list(SENT_COLORS.keys()),
                                range=list(SENT_COLORS.values())),
                legend=alt.Legend(title=None)),
            tooltip=["sentiment","n"])
        st.altair_chart(chart, use_container_width=True)
    with b:
        st.markdown("**Comments by survey question**")
        qc = view["source_question"].value_counts().reset_index()
        qc.columns = ["question","n"]
        st.altair_chart(alt.Chart(qc).mark_bar(color="#4C78A8").encode(
            x="n:Q", y=alt.Y("question:N", sort="-x", title=None),
            tooltip=["question","n"]), use_container_width=True)

    st.markdown("**Comments by site**")
    stc = view["site"].value_counts().reset_index()
    stc.columns = ["site","n"]
    st.altair_chart(alt.Chart(stc.head(20)).mark_bar(color="#4C78A8").encode(
        x=alt.X("n:Q", title="Comments"), y=alt.Y("site:N", sort="-x", title=None),
        tooltip=["site","n"]), use_container_width=True)

# ===========================================================================
# TAB — ACTION PRIORITIES
# ===========================================================================
with tab_prio:
    st.subheader("Action Priorities — where to focus")
    st.caption("Ranks each theme/category by combining how many people raised it, "
               "how negative they were, and the Tableau-spec **Friction** score. "
               "Respects the sidebar filters — filter to one site to get that site's list.")
    pc = st.columns([1, 2])
    lvl = pc[0].radio("Granularity", ["Categories", "Themes"], horizontal=True, key="prio_lvl")
    single_site = f_site[0] if len(f_site) == 1 else None
    if single_site:
        pc[1].info(f"Showing priorities for **{single_site}** (Friction is site-specific).")
    tfric = theme_friction_map(single_site)
    P = build_priority(fdf, lvl, tfric)

    if P.empty:
        st.info("No comments match the current filters.")
    else:
        show = P.head(15).copy()
        bar = alt.Chart(show).mark_bar().encode(
            x=alt.X("Priority:Q", title="Priority score (0–100)"),
            y=alt.Y("key:N", sort="-x", title=None),
            color=alt.Color("neg_share:Q", scale=alt.Scale(scheme="reds"),
                            legend=alt.Legend(title="% negative", format="%")),
            tooltip=[alt.Tooltip("key:N", title=lvl[:-1]),
                     alt.Tooltip("volume:Q", title="# comments"),
                     alt.Tooltip("neg_share:Q", title="% negative", format=".0%"),
                     alt.Tooltip("friction:Q", title="Friction (0–100)", format=".0f"),
                     alt.Tooltip("Priority:Q", format=".0f")])
        st.altair_chart(bar, use_container_width=True)

        disp = show.assign(**{
            "# comments": show["volume"],
            "% negative": (show["neg_share"] * 100).round(0).astype(int),
            "Friction": show["friction"].round(0),
            "priority": show["Priority"].round(0).astype(int),
        })[["key", "# comments", "% negative", "Friction", "priority"]] \
            .rename(columns={"key": lvl[:-1]})
        st.dataframe(disp, hide_index=True, use_container_width=True)
        st.caption("Priority = 40% comment volume + 35% share negative + 25% Friction "
                   "(each rescaled 0–1). Friction is the Tableau-spec under-support score (0–100); "
                   "themes with no matching rating item use the median. Org-wide unless one site is selected.")

        st.markdown("##### 🗣️ Representative comments")
        pick = st.selectbox(f"Show negative examples for:", show["key"].tolist(), key="prio_pick")
        ex = exemplars(fdf, pick, lvl)
        if len(ex):
            for _, r in ex.iterrows():
                st.markdown(f"> {str(r['comment'])}\n>\n> — *{r['site']} · {r['role']} · {r['source_question']}*")
        else:
            st.write("No strongly-negative example comments for this selection under the current filters.")

# ===========================================================================
# TAB — GAP ANALYSIS
# ===========================================================================
RYG_COLORS = RYG_HEX  # match the official Tableau legend (maroon / green / orange)

with tab_gap:
    st.subheader("Scores & Friction — matches the Tableau dashboard")
    st.caption("**Composite** = average Support on a 0–100 scale (Importance not factored). "
               "**Friction** = each respondent's shortfall of Support vs Importance, floored at 0, then averaged — "
               "so over-supported areas can't mask under-supported ones. Scope per the official spec: "
               "completed responses across the 10 in-scope sites.")

    if scores.empty:
        st.info("Scores not available — rebuild with build_data.py.")
    else:
        dimr = scores_dim[~scores_dim["dimension"].astype(str).str.startswith("OVERALL")]
        ow = scores_dim[scores_dim["dimension"] == "OVERALL (weighted)"]
        mcols = st.columns(len(dimr) + 1)
        for i, (_, r) in enumerate(dimr.iterrows()):
            mcols[i].metric(r["dimension"], f"{r['composite']:.0f}",
                            help=f"Composite (avg support, 0–100) · weight {int(r['weight'])}%")
        if len(ow):
            mcols[-1].metric("Overall (weighted)", f"{ow['composite'].iloc[0]:.0f}",
                             help="Work 35 · Service 25 · Physical 20 · Digital & Tech 20")

        st.divider()
        gsel1, gsel2 = st.columns(2)
        dsel = gsel1.selectbox("Dimension", ["All dimensions"] + dimr["dimension"].tolist())
        ssel = gsel2.selectbox("Site view", ["All 10 sites (overall)"] + meta.get("ten_sites", []))
        src = scores if ssel.startswith("All 10") else scores_site[scores_site["site"] == ssel]
        sd = src if dsel == "All dimensions" else src[src["dimension"] == dsel]
        sd = sd.dropna(subset=["friction"]).sort_values("friction", ascending=False).copy()
        sd["ryg"] = sd["ryg"].fillna("Agreement-only").replace("", "Agreement-only")

        if len(sd) == 0:
            st.info("No data for this selection.")
        else:
            st.markdown("**Friction by item** (0–100, higher = more under-supported)")
            bar = alt.Chart(sd).mark_bar().encode(
                x=alt.X("friction:Q", title="Friction (0–100)"),
                y=alt.Y("item:N", sort="-x", title=None),
                color=alt.Color("ryg:N", scale=alt.Scale(domain=list(RYG_COLORS), range=list(RYG_COLORS.values())),
                                legend=alt.Legend(title="Support vs Importance")),
                tooltip=["dimension", "item",
                         alt.Tooltip("composite:Q", title="Composite (support)", format=".0f"),
                         alt.Tooltip("friction:Q", format=".0f"),
                         alt.Tooltip("friction_pct:Q", title="% feeling friction", format=".0f"),
                         "ryg", alt.Tooltip("n:Q")])
            st.altair_chart(bar, use_container_width=True)
            st.caption("Digital & Tech has no Importance rating, so its Friction = 100 − Support "
                       "(every tech item shows some friction unless respondents fully agree) — not directly "
                       "comparable to the other three dimensions.")

            sca = sd.dropna(subset=["importance_100", "support_100"])
            if len(sca):
                xmid = float(scores["support_100"].dropna().mean())
                ymid = float(scores["importance_100"].dropna().mean())
                st.markdown(f"**Importance vs. Support quadrant** (0–100; crosshairs = org avg "
                            f"support {xmid:.0f} / importance {ymid:.0f})")
                pts = alt.Chart(sca).mark_circle(size=140, opacity=0.85).encode(
                    x=alt.X("support_100:Q", title="Support / satisfaction (0–100)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("importance_100:Q", title="Importance (0–100)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("ryg:N", scale=alt.Scale(domain=list(RYG_COLORS), range=list(RYG_COLORS.values())), legend=None),
                    tooltip=["item", alt.Tooltip("importance_100:Q", format=".0f"),
                             alt.Tooltip("support_100:Q", format=".0f"),
                             alt.Tooltip("friction:Q", format=".0f"), "ryg"])
                vline = alt.Chart(pd.DataFrame({"x": [xmid]})).mark_rule(color="#bbb", strokeDash=[4, 4]).encode(x="x")
                hline = alt.Chart(pd.DataFrame({"y": [ymid]})).mark_rule(color="#bbb", strokeDash=[4, 4]).encode(y="y")
                st.altair_chart(vline + hline + pts, use_container_width=True)
                st.caption("High importance + low support (upper-left of the crosshair) = *Immediate Attention*.")

            st.divider()
            st.markdown("### 🔗 Why? — comments behind an item")
            item_sel = st.selectbox("Pick an item to see the comments that explain its Friction", sd["item"].tolist())
            row = sd[sd["item"] == item_sel].iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Composite (support)", f"{row['composite']:.0f}" if pd.notna(row["composite"]) else "—")
            m2.metric("Friction", f"{row['friction']:.0f}")
            m3.metric("% feeling friction", f"{row['friction_pct']:.0f}%" if pd.notna(row["friction_pct"]) else "—")
            m4.metric("Status", row["ryg"] if row["ryg"] else "n/a (Tech)")

            rel_themes = themes_for_item(item_sel)
            st.caption("Related comment themes: " + (", ".join(rel_themes) if rel_themes else "—"))
            link = comments[~comments["eff_junk"]].copy()
            if not ssel.startswith("All 10"):
                link = link[link["site"] == ssel]
            if rel_themes:
                pat = "|".join([t.replace("(", "\\(").replace(")", "\\)").replace("/", "\\/") for t in rel_themes])
                link = link[link["eff_themes"].str.contains(pat, case=False, na=False, regex=True)]
            if len(link):
                sc = link["eff_sentiment"].value_counts()
                st.write(f"**{len(link):,} related comments** · "
                         f"🟢 {sc.get('Positive',0)}  ⚪ {sc.get('Neutral / Mixed',0)}  🔴 {sc.get('Negative',0)}")
                sent_pick = st.radio("Show", ["All", "Negative", "Positive", "Neutral / Mixed"],
                                     horizontal=True, key="gap_sent")
                showc = link if sent_pick == "All" else link[link["eff_sentiment"] == sent_pick]
                showc = showc.sort_values("sentiment_score")
                st.dataframe(showc[["site", "role", "source_question", "eff_sentiment", "eff_themes", "comment"]]
                             .rename(columns={"eff_sentiment": "sentiment", "eff_themes": "themes"}),
                             use_container_width=True, height=360, hide_index=True)
            else:
                st.info("No matching comments for this item / site.")

# ===========================================================================
# TAB 3 — THEME & SENTIMENT EXPLORER
# ===========================================================================
with tab_expl:
    st.subheader("Theme & Sentiment Explorer")
    st.caption("Reflects the sidebar filters. Multi-topic comments count under each theme/category they mention.")
    if len(fdf) == 0:
        st.info("No comments match the current filters.")
    else:
        cc = st.columns(3)
        level = cc[0].radio("View by", ["Categories", "Themes"], horizontal=True, key="expl_level")
        breakdown = cc[1].selectbox("Break down by", ["Site", "Role", "Tenure", "Hybrid category"], key="expl_break")
        metric = cc[2].radio("Heatmap metric", ["% of group", "Counts"], horizontal=True, key="expl_metric")
        BRK = {"Site": "site", "Role": "role", "Tenure": "tenure", "Hybrid category": "hybrid"}[breakdown]

        base = fdf[~fdf["eff_junk"]] if not inc_junk else fdf
        tex = explode_level(base, level)
        noun = level.lower()

        left, right = st.columns([1,1])
        with left:
            st.markdown(f"**Top {noun}**")
            tc = tex["key"].value_counts().head(20).reset_index(); tc.columns = ["key","n"]
            st.altair_chart(alt.Chart(tc).mark_bar(color="#4C78A8").encode(
                x=alt.X("n:Q", title="Mentions"), y=alt.Y("key:N", sort="-x", title=None),
                tooltip=["key","n"]), use_container_width=True)
        with right:
            st.markdown(f"**Sentiment mix (top 12 {noun})**")
            top12 = tex["key"].value_counts().head(12).index.tolist()
            sb = tex[tex["key"].isin(top12)]
            grp = sb.groupby(["key","eff_sentiment"]).size().reset_index(name="n")
            st.altair_chart(alt.Chart(grp).mark_bar().encode(
                x=alt.X("n:Q", stack="normalize", title="Share"),
                y=alt.Y("key:N", sort=top12, title=None),
                color=alt.Color("eff_sentiment:N",
                    scale=alt.Scale(domain=list(SENT_COLORS.keys()), range=list(SENT_COLORS.values())),
                    legend=alt.Legend(title="Sentiment", orient="bottom")),
                tooltip=["key","eff_sentiment","n"]), use_container_width=True)

        unit = f"% of each {breakdown.lower()}'s comments" if metric == "% of group" else "mention counts"
        st.markdown(f"**{level} × {breakdown}** ({unit})")
        top_k = tex["key"].value_counts().head(15).index.tolist()
        top_b = base[BRK].value_counts().head(12).index.tolist()
        hm = tex[tex["key"].isin(top_k) & tex[BRK].isin(top_b)]
        hm = hm.groupby([BRK,"key"]).size().reset_index(name="n")
        if metric == "% of group":
            den = base[base[BRK].isin(top_b)].groupby(BRK)["comment_id"].nunique().rename("den")
            hm = hm.merge(den, left_on=BRK, right_index=True)
            hm["val"] = hm["n"] / hm["den"] * 100
            ctitle, cfmt = "% of group", ".0f"
        else:
            hm["val"] = hm["n"]; ctitle, cfmt = "Mentions", "d"
        st.altair_chart(alt.Chart(hm).mark_rect().encode(
            x=alt.X(f"{BRK}:N", title=None, sort=top_b),
            y=alt.Y("key:N", title=None, sort=top_k),
            color=alt.Color("val:Q", scale=alt.Scale(scheme="blues"), legend=alt.Legend(title=ctitle)),
            tooltip=[alt.Tooltip(f"{BRK}:N", title=breakdown), alt.Tooltip("key:N", title=level[:-1]),
                     alt.Tooltip("val:Q", title=ctitle, format=cfmt),
                     alt.Tooltip("n:Q", title="mentions")]), use_container_width=True)
        if metric == "% of group":
            st.caption("Normalized so small sites/segments are comparable to large ones — "
                       "each cell is the share of that group's comments mentioning the theme, not raw volume.")

        st.divider()
        st.markdown(f"**Comments ({len(fdf):,})**")
        st.dataframe(
            fdf[["site","role","tenure","source_question","eff_sentiment","eff_themes","comment"]]
               .rename(columns={"eff_sentiment":"sentiment","eff_themes":"themes"}),
            use_container_width=True, height=420, hide_index=True)
        st.download_button("⬇️ Download filtered comments (CSV)",
            fdf.to_csv(index=False).encode(), "pulsepoint_filtered.csv", "text/csv")

# ===========================================================================
# TAB — BROWSE & PIVOT
# ===========================================================================
with tab_browse:
    st.subheader("Browse & Pivot")
    b_browse, b_pivot, b_export = st.tabs(["🗂️ Comment browser", "📐 Pivot builder", "⬇️ Exports"])

    with b_browse:
        st.caption("Every comment, color-coded by sentiment with theme tags. Reflects the sidebar filters "
                   "(site, role, sentiment, theme, keyword).")
        view = fdf[~fdf["eff_junk"]] if not inc_junk else fdf
        cA, cB, cC = st.columns(3)
        cA.metric("Comments", f"{len(view):,}")
        cB.metric("% negative", f"{(view['eff_sentiment']=='Negative').mean()*100:.0f}%" if len(view) else "0%")
        cC.metric("% positive", f"{(view['eff_sentiment']=='Positive').mean()*100:.0f}%" if len(view) else "0%")
        show = view[["eff_sentiment", "eff_themes", "comment", "site", "role", "tenure",
                     "source_question", "sentiment_score"]].rename(
            columns={"eff_sentiment": "sentiment", "eff_themes": "themes", "source_question": "question"})
        sty = show.head(500).style.apply(
            lambda col: [f"background-color: {SENT_COLORS.get(v, '#fff')}33;"
                         f"color:#111;font-weight:600" for v in col]
            if col.name == "sentiment" else ["" for _ in col], axis=0)
        st.dataframe(sty, use_container_width=True, height=520, hide_index=True)
        if len(view) > 500:
            st.caption(f"Showing first 500 of {len(view):,}. Narrow the sidebar filters to see the rest, "
                       "or use the offline dashboard in the Exports tab.")

    with b_pivot:
        st.caption("Cross-tab any two dimensions. Multi-topic comments are counted under each theme they mention.")
        DIMS = {"Site": "site", "Role": "role", "Tenure": "tenure", "Hybrid": "hybrid",
                "Theme": "__theme", "Category": "__category", "Sentiment": "eff_sentiment",
                "Question": "source_question"}
        pc = st.columns(3)
        rowdim = pc[0].selectbox("Rows", list(DIMS), index=4, key="pv_row")
        coldim = pc[1].selectbox("Columns", list(DIMS), index=0, key="pv_col")
        metric = pc[2].selectbox("Value", ["Count", "% of row", "% of column", "Avg sentiment score"], key="pv_met")
        base = (fdf[~fdf["eff_junk"]] if not inc_junk else fdf).copy()
        need_explode = DIMS[rowdim] in ("__theme", "__category") or DIMS[coldim] in ("__theme", "__category")
        if need_explode:
            base = base.assign(__k=base["eff_themes"].str.split("|")).explode("__k")
            base["__k"] = base["__k"].str.strip()
            base = base[(base["__k"] != "") & (base["__k"] != "Unclassified")]
            base["__theme"] = base["__k"]
            base["__category"] = base["__k"].map(lambda x: THEME_TO_CAT.get(x, "Uncategorized"))
        rd, cd = DIMS[rowdim], DIMS[coldim]
        if rd == cd:
            st.info("Pick two different dimensions.")
        elif len(base) == 0:
            st.info("No comments match the current filters.")
        else:
            if metric == "Avg sentiment score":
                pv = base.pivot_table(index=rd, columns=cd, values="sentiment_score", aggfunc="mean")
                fmt = "{:.2f}"
            else:
                pv = base.pivot_table(index=rd, columns=cd, values="comment_id",
                                      aggfunc="nunique", fill_value=0)
                if metric == "% of row":
                    pv = pv.div(pv.sum(axis=1), axis=0) * 100; fmt = "{:.0f}%"
                elif metric == "% of column":
                    pv = pv.div(pv.sum(axis=0), axis=1) * 100; fmt = "{:.0f}%"
                else:
                    fmt = "{:.0f}"
            pv = pv.loc[pv.sum(axis=1).sort_values(ascending=False).index]
            sty = pv.style.format(fmt).background_gradient(
                cmap="Blues" if metric != "Avg sentiment score" else "RdYlGn", axis=None)
            st.dataframe(sty, use_container_width=True, height=460)
            st.download_button("⬇️ Download this pivot (CSV)", pv.to_csv().encode(),
                               "pulsepoint_pivot.csv", "text/csv")

    with b_export:
        st.caption("Take the analysis with you — no Streamlit required.")
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**📊 Excel analysis workbook**")
            st.caption("Comments, theme summary, site×theme and sentiment×theme pivots, verbatims, and gaps — "
                       "each on its own sheet for slicing in Excel.")
            st.download_button("⬇️ Download workbook (.xlsx)", build_excel_workbook(completers_only),
                               "pulsepoint_analysis.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
        with e2:
            st.markdown("**🌐 No-install dashboard**")
            st.caption("A single self-contained HTML file with filters, live theme bars, and a searchable "
                       "comment browser. Double-click to open in any browser — no Python needed.")
            st.download_button("⬇️ Download dashboard (.html)", build_standalone_dashboard(completers_only),
                               "pulsepoint_dashboard.html", "text/html", use_container_width=True)

# ===========================================================================
# TAB — HOTSPOTS & KEY DRIVERS
# ===========================================================================
with tab_diag:
    st.subheader("Hotspots & Key Drivers")
    st.caption("Org-wide diagnostics — these use all responses and ignore the sidebar filters.")
    h_tab, d_tab = st.tabs(["🔥 Hotspots", "📈 Key drivers of satisfaction"])

    with h_tab:
        st.markdown("**Where is a theme *unusually* common?**")
        st.caption("Flags site/role/tenure groups whose mention rate for a theme is significantly "
                   "above the rest of the org (two-proportion z-test, p < 0.05). Small groups are "
                   "held to the same significance bar, so you're not chasing noise.")
        hc = st.columns([1, 1])
        dimlabel = hc[0].radio("Compare across", ["Site", "Role", "Tenure"], horizontal=True, key="hot_dim")
        dim = {"Site": "site", "Role": "role", "Tenure": "tenure"}[dimlabel]
        min_lift = hc[1].slider("Minimum lift vs org rate", 1.10, 3.0, 1.25, 0.05, key="hot_lift")
        H = build_hotspots(comments, dim, min_lift=min_lift)
        if H.empty:
            st.info("No significant hotspots at these settings — try lowering the minimum lift.")
        else:
            disp = H.rename(columns={dim: dimlabel, "group_rate": "group %", "org_rate": "org %",
                                     "group_n": "group size", "mentions": "# mentions"})
            chart = alt.Chart(H.head(20)).mark_circle().encode(
                x=alt.X("lift:Q", title="Lift vs org (×)"),
                y=alt.Y("theme:N", sort="-x", title=None),
                size=alt.Size("mentions:Q", legend=alt.Legend(title="# mentions")),
                color=alt.Color(f"{dim}:N", legend=alt.Legend(title=dimlabel)),
                tooltip=[alt.Tooltip(f"{dim}:N", title=dimlabel), "theme",
                         alt.Tooltip("group_rate:Q", title="group %"),
                         alt.Tooltip("org_rate:Q", title="org %"),
                         alt.Tooltip("lift:Q"), alt.Tooltip("mentions:Q"),
                         alt.Tooltip("p:Q", format=".3f")])
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(disp[[dimlabel, "theme", "group %", "org %", "lift", "# mentions", "group size", "p"]],
                         hide_index=True, use_container_width=True)
            st.caption(f"{len(H)} significant hotspots. lift = group rate ÷ org rate. "
                       "Example reading: a 1.4× lift means that group raises the theme 40% more often than the org.")

    with d_tab:
        st.markdown("**Which themes actually move overall satisfaction?**")
        st.caption("Uses the survey's own overall-agreement scores (1–5). For each theme, the average "
                   "score of people who raised it vs everyone else. A negative delta means raising the "
                   "theme goes with *lower* overall satisfaction; 'impact' weights that by how many "
                   "people it affects. This tells you which complaints are worth fixing first.")
        if resp_scores.empty:
            st.info("Overall-agreement scores weren't found in the source data.")
        else:
            avail = [k for k, v in SCORE_LABELS.items()
                     if v == "composite" or v in meta.get("score_cols", [])]
            outsel = st.selectbox("Outcome measure", avail, key="drv_out")
            D = build_driver(comments, resp_scores, SCORE_LABELS[outsel])
            if D.empty:
                st.info("Not enough data for this outcome.")
            else:
                worst = D.head(12)
                bar = alt.Chart(worst).mark_bar().encode(
                    x=alt.X("delta:Q", title="Satisfaction delta (raised theme − did not)"),
                    y=alt.Y("theme:N", sort="x", title=None),
                    color=alt.Color("delta:Q", scale=alt.Scale(scheme="redblue", domainMid=0), legend=None),
                    tooltip=["theme", alt.Tooltip("n:Q", title="# people"),
                             alt.Tooltip("sat_with:Q", title="score if raised"),
                             alt.Tooltip("sat_without:Q", title="score if not"),
                             alt.Tooltip("delta:Q"), alt.Tooltip("impact:Q"),
                             alt.Tooltip("p:Q", format=".3f")])
                st.altair_chart(bar, use_container_width=True)
                disp = D.assign(**{"significant (p<.05)": D["p"] < 0.05})
                disp = disp.rename(columns={"n": "# people", "sat_with": "score if raised",
                                            "sat_without": "score if not"})
                st.dataframe(disp[["theme", "# people", "score if raised", "score if not",
                                   "delta", "impact", "significant (p<.05)"]].sort_values("impact"),
                             hide_index=True, use_container_width=True)
                st.caption("Most-negative impact = biggest drag on overall satisfaction. Themes with "
                           "near-zero, non-significant deltas (e.g. reception, community events) barely "
                           "move the needle — useful to know before investing there.")

# ===========================================================================
# TAB — SITE REPORTS (one-pager export)
# ===========================================================================
with tab_report:
    st.subheader("📋 10 PulsePoints — Portfolio report card")
    st.caption("A company-wide exec brief across the 10 in-scope sites: an Index-Score leaderboard, a site×dimension "
               "heatmap, Importance-vs-Support gaps, company Friction & strengths, top comment themes with a "
               "site×theme heatmap, and the respondent profile. Download as HTML or PDF.")
    pf_html = build_portfolio_report(completers_only)
    _pcol = st.columns(4)
    _pcol[0].download_button("⬇️ Portfolio HTML", pf_html, "pulsepoint_portfolio.html", "text/html",
                             use_container_width=True)
    _safe_pdf_button(_pcol[1], "⬇️ Portfolio PDF", lambda: build_portfolio_pdf(completers_only),
                     "pulsepoint_portfolio.pdf", key="pf_pdf")
    components.html(pf_html, height=780, scrolling=True)
    st.divider()

    st.subheader("Site one-pagers")
    st.caption("A leadership-ready summary for a single site — top priorities, highest-Friction items, "
               "org benchmark, and representative quotes, with an auto-written recap. The 10 in-scope "
               "sites (per the Tableau spec) carry official Friction/Composite scores; others show the "
               "comment analysis only. Download as HTML, PDF, or PowerPoint.")
    report_sites = meta.get("ten_sites", meta["sites"])
    rc = st.columns([2, 1])
    _order = comments[comments["site"].isin(report_sites) & ~comments["eff_junk"]]["site"].value_counts()
    _default = _order.index[0] if len(_order) else report_sites[0]
    _idx = report_sites.index(_default) if _default in report_sites else 0
    rsite = rc[0].selectbox("Site (10 in-scope, defaults to largest)", report_sites, index=_idx, key="rep_site")
    safe = rsite.replace("/", "-")
    report_html = build_site_report(rsite)
    with rc[1]:
        st.download_button("⬇️ HTML", report_html, f"pulsepoint_{safe}.html", "text/html",
                           use_container_width=True)
        _safe_pdf_button(st, "⬇️ PDF", lambda: build_site_pdf(rsite, completers_only),
                         f"pulsepoint_{safe}.pdf", key="site_pdf")
        st.download_button("⬇️ PowerPoint", build_site_pptx([rsite], completers_only), f"pulsepoint_{safe}.pptx",
                           "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                           use_container_width=True)

    # --- Operational Action Plan — What to Do Next (generate / regenerate) -----------
    with st.expander("🧭 Operational Action Plan — generate / regenerate",
                     expanded=not action_plan.plan_exists(rsite)):
        st.caption("An auto-generated, validate-before-acting **What to Do Next** section that renders at the "
                   "top of the card. Grounded only in this site's evidence; generated once and stored.")
        _prov = action_plan.configured_provider()
        _pm = action_plan.plan_meta(rsite)
        if _pm:
            st.caption(f"Current plan for **{rsite}**: generated {_pm.get('generated_at','?')} · "
                       f"{_pm.get('provider','?')}/{_pm.get('model','?')}"
                       + (" · aggregate-only" if _pm.get('include_verbatims') is False else ""))
        else:
            st.caption(f"No action plan generated yet for **{rsite}** — it appears at the top of the card once created.")
        oc = st.columns([3, 2, 2])
        _owner = oc[0].text_input("Primary owner (every action)", value=action_plan.default_owner(),
                                  key="oap_owner", help="The ownership model is still open per the meeting — edit freely.")
        _incl = oc[1].checkbox("Include verbatim quotes", value=True, key="oap_incl",
                               help="On = quotes ground the 'what this looks like' lines. "
                                    "Off = aggregate-only; no comment text is sent to the model.")
        _btn_label = "🔁 Regenerate" if _pm else "✨ Generate"
        if oc[2].button(_btn_label, type="primary", use_container_width=True,
                        disabled=(_prov is None), key="oap_gen"):
            try:
                with st.spinner(f"Generating action plan for {rsite} via {_prov}…"):
                    _data = site_report_data(rsite, completers_only)
                    _plan = action_plan.generate_action_plan(_data, owner=_owner, include_verbatims=_incl)
                    action_plan.save_plan(rsite, _plan)
                    build_site_pdf.clear()
                st.success(f"Action plan generated for {rsite}.")
                st.rerun()
            except Exception as e:
                st.error(f"Generation failed: {e}")
        if _prov is None:
            st.info("⚙️ **No model configured.** Recommended (keeps comments on your machine): install "
                    "[Ollama](https://ollama.com), run `ollama pull llama3.1`, then set "
                    "`PULSEPOINT_LLM_PROVIDER=ollama` and restart the app. Azure / OpenAI / Anthropic are also "
                    "supported — see the README → *Operational Action Plan*.")
        elif st.button("Generate for all 10 sites", key="oap_gen_all",
                       help="Runs the model once per site and stores each plan."):
            _prog = st.progress(0.0, text="Starting…")
            _fail = []
            for _i, _s in enumerate(report_sites):
                try:
                    _d = site_report_data(_s, completers_only)
                    _pl = action_plan.generate_action_plan(_d, owner=_owner, include_verbatims=_incl)
                    action_plan.save_plan(_s, _pl)
                except Exception as e:
                    _fail.append(f"{_s}: {e}")
                _prog.progress((_i + 1) / len(report_sites), text=f"{_i+1}/{len(report_sites)} — {_s}")
            build_site_pdf.clear()
            (st.warning("Some sites failed:\n\n" + "\n\n".join(_fail)) if _fail
             else st.success(f"Generated action plans for all {len(report_sites)} sites."))
            st.rerun()

    components.html(report_html, height=820, scrolling=True)

    st.divider()
    st.markdown("**Bulk exports & verbatims**")
    bc = st.columns(3)
    if bc[0].button("📦 All 10 sites — HTML (zip)", use_container_width=True):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for s in report_sites:
                z.writestr(f"pulsepoint_{s.replace('/', '-')}.html", build_site_report(s))
        st.download_button("⬇️ Download HTML zip", buf.getvalue(),
                           "pulsepoint_site_reports.zip", "application/zip")
    if bc[1].button("📊 All 10 sites — PowerPoint deck", use_container_width=True):
        st.download_button("⬇️ Download deck (.pptx)", build_site_pptx(report_sites, completers_only),
                           "pulsepoint_all_sites.pptx",
                           "application/vnd.openxmlformats-officedocument.presentationml.presentation")
    with bc[2]:
        vb = verbatim_export(rsite)
        st.download_button(f"🗒️ {rsite} verbatims (CSV)", vb.to_csv(index=False).encode(),
                           f"verbatims_{safe}.csv", "text/csv", use_container_width=True)

# ===========================================================================
# TAB 4 — TAGGING WORKSPACE
# ===========================================================================
with tab_tag:
    st.subheader("Tagging Workspace")
    st.caption("Review auto-tags and override them. Edit **themes_override**, **sentiment_override** "
               "(Positive / Negative / Neutral / Mixed), **junk_override** (True/False), add **notes**, "
               "tick **reviewed**. Save writes to data/overrides.csv and updates every tab.")

    with st.expander(f"🔍 Discover missing themes — {len(emerging)} clusters found in the "
                     f"{int((comments['eff_themes']=='Unclassified').sum())} Unclassified comments"):
        st.caption("These are auto-clustered from comments the taxonomy couldn't tag. Each cluster is a "
                   "candidate new theme — if one looks real, add its keywords to `taxonomy.py` and rebuild, "
                   "or tag its comments by hand below.")
        if not emerging:
            st.write("No sizable clusters — the taxonomy is covering the comments well.")
        for i, cl in enumerate(emerging, 1):
            st.markdown(f"**Cluster {i} · {cl['size']} comments** — keywords: "
                        f"*{', '.join(cl['keywords'])}*")
            for s in cl["samples"][:3]:
                st.markdown(f"> {_html.escape(str(s))}")

    # Focused subset for editing
    e1,e2,e3 = st.columns([1,1,1])
    only_unrev = e1.checkbox("Only unreviewed", value=False)
    only_unclass = e2.checkbox("Only Unclassified / junk-suspect", value=False)
    edit_kw = e3.text_input("Filter by keyword", "", key="edit_kw")

    edf = apply_filters(comments) if (f_site or f_role or f_hybrid or f_tenure or f_ques or f_sent or f_theme or kw) else comments.copy()
    if only_unrev:
        edf = edf[~edf["reviewed"].astype(str).isin(["True","true","1"])]
    if only_unclass:
        edf = edf[edf["eff_themes"].str.contains("Unclassified", na=False) | edf["eff_junk"]]
    if edit_kw.strip():
        edf = edf[edf["comment"].str.contains(edit_kw.strip(), case=False, na=False)]

    st.write(f"Editing **{len(edf):,}** comments "
             f"(showing auto-tags; blank override = keep auto).")

    edit_cols = ["comment_id","site","source_question","comment",
                 "sentiment_label","themes",
                 "sentiment_override","themes_override","junk_override","notes","reviewed"]
    edited = st.data_editor(
        edf[edit_cols].head(600),
        use_container_width=True, height=520, hide_index=True, num_rows="fixed",
        disabled=["comment_id","site","source_question","comment","sentiment_label","themes"],
        column_config={
            "comment": st.column_config.TextColumn("comment", width="large"),
            "sentiment_label": st.column_config.TextColumn("auto sentiment", width="small"),
            "themes": st.column_config.TextColumn("auto themes", width="medium"),
            "sentiment_override": st.column_config.SelectboxColumn(
                "sentiment override", options=["","Positive","Negative","Neutral / Mixed"], width="small"),
            "themes_override": st.column_config.TextColumn("themes override (pipe|separated)", width="medium"),
            "junk_override": st.column_config.SelectboxColumn(
                "junk?", options=["","True","False"], width="small"),
            "reviewed": st.column_config.CheckboxColumn("reviewed"),
        },
        key="editor")
    if len(edf) > 600:
        st.caption("Showing first 600 rows — narrow with filters above to edit the rest.")

    csave, cdl, cdl2 = st.columns([1,1,1])
    if csave.button("💾 Save changes", type="primary"):
        # load existing overrides, update with edited rows
        if os.path.exists(OVERRIDES):
            ov = pd.read_csv(OVERRIDES)
        else:
            ov = comments[["comment_id"]].copy()
            for c in ["themes_override","sentiment_override","junk_override","notes","reviewed"]:
                ov[c] = "" if c != "reviewed" else False
        ov = ov.set_index("comment_id")
        for _, r in edited.iterrows():
            cid = r["comment_id"]
            ov.loc[cid, "themes_override"] = r["themes_override"]
            ov.loc[cid, "sentiment_override"] = r["sentiment_override"]
            ov.loc[cid, "junk_override"] = r["junk_override"]
            ov.loc[cid, "notes"] = r["notes"]
            ov.loc[cid, "reviewed"] = bool(r["reviewed"])
        ov.reset_index().to_csv(OVERRIDES, index=False)
        st.cache_data.clear()
        st.success(f"Saved {len(edited)} rows to overrides.csv. Reloading…")
        st.rerun()

    # export the full merged dataset (auto + overrides applied)
    full_export = comments[["comment_id","respondent","site","role","hybrid","tenure",
        "source_question","comment","eff_junk","eff_sentiment","eff_themes",
        "sentiment_label","themes","sentiment_override","themes_override","junk_override",
        "notes","reviewed"]].rename(columns={
        "eff_junk":"final_junk","eff_sentiment":"final_sentiment","eff_themes":"final_themes"})
    cdl.download_button("⬇️ Export tagged dataset (CSV)",
        full_export.to_csv(index=False).encode(), "pulsepoint_tagged.csv", "text/csv")

    @st.cache_data
    def to_xlsx(df):
        from io import BytesIO
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="Tagged comments")
            if not scores.empty:
                scores.to_excel(w, index=False, sheet_name="Scores & Friction")
        return buf.getvalue()
    cdl2.download_button("⬇️ Export to Excel (comments + scores)",
        to_xlsx(full_export), "pulsepoint_analysis.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
