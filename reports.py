"""
PulsePoint report core — data loading, computation and HTML/PDF/PPTX builders.

Importable without Streamlit. Everything that depends on the selected population
(completed responses only vs all) reads it from an explicit ``Ctx`` object built
by :func:`make_context`, so the builders can be called from tests and scripts:

    import reports as R
    ctx = R.load_context(completers_only=True)
    html = R.build_site_report(ctx, "IN-Indianapolis")

``app.py`` keeps the Streamlit UI and wraps the expensive calls in st.cache_data.
"""
import os, json, re, html as _html, io, base64
from collections import defaultdict
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import action_plan  # Operational Action Plan — "What to Do Next" section (LLM-generated, stored per site)
from taxonomy import THEME_TO_CATEGORY, GENERIC_THEMES

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OVERRIDES = os.path.join(DATA, "overrides.csv")

# theme -> category (Unclassified rolls up to Uncategorized), same as meta["theme_to_category"]
THEME_TO_CAT = {**THEME_TO_CATEGORY, "Unclassified": "Uncategorized"}


@dataclass
class Ctx:
    """The data a report is built from, for one population (completed-only or all)."""
    comments: pd.DataFrame
    resp_scores: pd.DataFrame
    scores: pd.DataFrame
    scores_site: pd.DataFrame
    scores_dim: pd.DataFrame
    meta: dict
    completers_only: bool
    theme_friction_all: dict = field(default_factory=dict)


def make_context(comments_full, resp_scores_full, scores, meta, completers_only):
    """Select the population and bundle everything the builders read."""
    k = "comp" if completers_only else "all"
    if completers_only:
        c = comments_full[comments_full["reached_end"] == 1].copy()
        r = resp_scores_full[resp_scores_full["reached_end"] == 1].copy()
    else:
        c, r = comments_full.copy(), resp_scores_full.copy()
    ctx = Ctx(comments=c, resp_scores=r, scores=scores[k], scores_site=scores[k + "_site"],
              scores_dim=scores[k + "_dim"], meta=meta, completers_only=bool(completers_only))
    ctx.theme_friction_all = _theme_friction_from(ctx.scores)
    return ctx


def load_context(completers_only=True):
    """Convenience for scripts/tests: load everything from data/ and build a Ctx."""
    comments_full, meta, _gaps = load_all()
    resp_scores_full, _emg = load_extra()
    return make_context(comments_full, resp_scores_full, load_scores(), meta, completers_only)

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

def theme_friction_map(ctx, site=None):
    """Theme->Friction (0-100). Site-specific when a single site is selected."""
    scores_site, THEME_FRICTION_ALL = ctx.scores_site, ctx.theme_friction_all
    if site and not scores_site.empty and site in set(scores_site["site"]):
        m = _theme_friction_from(scores_site[scores_site["site"] == site])
        return {**THEME_FRICTION_ALL, **m}
    return THEME_FRICTION_ALL

def category_friction_map(tfric):
    tmp = defaultdict(list)
    for th, g in tfric.items():
        tmp[THEME_TO_CAT.get(th, "Uncategorized")].append(g)
    return {c: float(np.mean(v)) for c, v in tmp.items()}

def explode_level(df, level, exclude_generic=False):
    """Explode eff_themes to one row per theme; roll up to category if requested.
    exclude_generic drops taxonomy.GENERIC_THEMES before any roll-up — use it
    wherever themes are ranked or rolled up for priorities."""
    t = df.assign(_k=df["eff_themes"].str.split("|")).explode("_k")
    t["_k"] = t["_k"].str.strip()
    t = t[(t["_k"] != "") & (t["_k"] != "Unclassified")]
    if exclude_generic:
        t = t[~t["_k"].isin(GENERIC_THEMES)]
    if level == "Categories":
        t["_k"] = t["_k"].map(lambda x: THEME_TO_CAT.get(x, "Uncategorized"))
        t = t[t["_k"] != "Uncategorized"]
    # reset index: explode leaves duplicate labels, which break crosstab/alignment
    return t.rename(columns={"_k": "key"}).reset_index(drop=True)

def build_priority(df, level, tfric):
    real = df[~df["eff_junk"]]
    if real.empty:
        return pd.DataFrame()
    ex = explode_level(real, level, exclude_generic=True)
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
            lambda s: any(THEME_TO_CAT.get(t.strip()) == key for t in str(s).split("|")
                          if t.strip() not in GENERIC_THEMES))
    else:
        mask = real["eff_themes"].str.contains(re.escape(key), na=False)
    sub = real[mask & (real["eff_sentiment"] == "Negative")]
    sub = sub[(sub["char_len"] >= 40) & (sub["char_len"] <= 280)]
    return sub.sort_values("sentiment_score").head(n)

# ---------------------------------------------------------------------------
# Extra data: respondent-level scores (key drivers) + emerging clusters
# ---------------------------------------------------------------------------
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

SCORE_LABELS = {"Overall (all sections)": "composite", "Workspace support": "score_work",
                "Physical space": "score_space", "Services": "score_services",
                "Technology": "score_tech"}

def build_hotspots(comments_df, dim="site", min_n=40, min_lift=1.15, alpha=0.05):
    """Theme rates in a group significantly above the rest (two-proportion z-test)."""
    from scipy.stats import norm
    real = comments_df[~comments_df["eff_junk"]]
    ex = explode_level(real, "Themes", exclude_generic=True)
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
        lambda s: set(t for t in "|".join(s).split("|")
                      if t and t != "Unclassified" and t not in GENERIC_THEMES))

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
            lambda s: any(THEME_TO_CAT.get(t.strip()) == key for t in str(s).split("|")
                          if t.strip() not in GENERIC_THEMES))
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

def item_status(r):
    """Official status for one item row: the stored `ryg` column from build_data.py
    (computed on unrounded means). This is the single source for item-level status —
    never recompute it from a rounded gap. Agreement-only items return "Agreement-only"."""
    v = r.get("ryg") if hasattr(r, "get") else None
    if isinstance(v, str) and v.strip():
        return v.strip()
    return "Agreement-only"


def ryg_of(gap5):
    """RYG status from Support − Importance (1–5), per the Tableau spec thresholds.
    Use only for DIMENSION-level averages, which have no stored status; item rows
    must use item_status()."""
    if pd.isna(gap5):
        return "Agreement-only"
    if gap5 < -0.049:
        return "Under Supported"
    if gap5 > 1.5:
        return "Over Supported"
    return "Target"

# "What's working" rule (site card + action-plan payload)
STRENGTH_MIN_COMMENTS = 10     # a theme needs at least this many comments at the site
STRENGTH_MIN_POS_SHARE = 0.50  # ... a positive share of 50% or more
STRENGTH_MIN_THEMES = 2        # fewer qualifying themes -> fill from best-supported items

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

def site_standings(ctx):
    """Per-site overall Index Score (unweighted mean of the four dimension composites,
    0–100 — this is what matches the Tableau deck) + per-dimension composites, for ranking."""
    ss = ctx.scores_site
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

def agreement_sig(ctx, site, col, ten_sites):
    """Welch's t-test of a site's 1–5 summary-agreement scores vs the OTHER in-scope
    sites. Returns site/rest means, delta, p, a plain-language verdict, and a color.
    This is the defensible 'raw number + a statistical thing' for the overall grade."""
    resp_scores = ctx.resp_scores
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

def site_report_data(ctx, site):
    """All computed pieces for a site's report."""
    comments, scores_site, resp_scores = ctx.comments, ctx.scores_site, ctx.resp_scores
    scores, scores_dim, meta = ctx.scores, ctx.scores_dim, ctx.meta
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
    data["P"] = build_priority(sc, "Categories", theme_friction_map(ctx, site)).head(5)
    # Generic buckets (taxonomy.GENERIC_THEMES) are hidden from every theme ranking on
    # the card; their comments still count toward sentiment and the comment appendix.
    tex = explode_level(real, "Themes", exclude_generic=True)
    tcounts = tex["key"].value_counts()
    data["top_themes"] = tcounts.head(8)
    # theme sentiment for strengths/fixes
    tsent = tex.groupby("key")["eff_sentiment"].apply(lambda s: (s == "Negative").mean())
    tpos = tex.groupby("key")["eff_sentiment"].apply(lambda s: (s == "Positive").mean())
    tvol = tcounts
    fixes = [(t, int(tvol[t]), float(tsent[t])) for t in tvol.index
             if tvol[t] >= 5 and tsent.get(t, 0) >= 0.45][:5]
    # What's working: the TRUE positive share (not 1 − negative). A theme qualifies with
    # enough comments, a positive majority, and more positive than negative comments.
    strengths = [(t, int(tvol[t]), float(tpos.get(t, 0.0))) for t in tvol.index
                 if tvol[t] >= STRENGTH_MIN_COMMENTS
                 and tpos.get(t, 0.0) >= STRENGTH_MIN_POS_SHARE
                 and tpos.get(t, 0.0) > tsent.get(t, 1.0)][:5]
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
    standings = site_standings(ctx)
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
    # If comments give fewer than two clear strengths, fill "What's working" from the
    # best-supported survey items — labelled as ratings, never as comment shares.
    data["strength_items"] = []
    if len(strengths) < STRENGTH_MIN_THEMES and len(imp_ss):
        need = STRENGTH_MIN_THEMES - len(strengths)
        for _, r in imp_ss.sort_values("support_100", ascending=False).head(need).iterrows():
            data["strength_items"].append({"item": str(r["item"]), "dimension": str(r["dimension"]),
                                           "support_100": float(r["support_100"])})
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
        sig = agreement_sig(ctx, site, DIM_TO_SCORECOL[dm], ten)
        if sig is not None:
            agreement.append({"dim": dm, "q": AGREE_Q[dm], **sig})
    data["agreement"] = agreement
    data["sentiment_overall"] = agreement_sig(ctx, site, "composite", ten)
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
            # how many negative comments carry this theme (the quote is one of them) —
            # a count of the pool it was drawn from, not a claim that they say the same
            n_neg = int(((tex["key"] == t) & (tex["eff_sentiment"] == "Negative")).sum())
            quotes.append({"theme": t, "text": str(r["comment"]), "role": str(r["role"]),
                           "tenure": str(r["tenure"]), "sentiment": r["eff_sentiment"],
                           "n_negative_in_theme": n_neg})
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
    # Extra evidence for the Operational Action Plan generator only (not rendered on the
    # card): a top-5 plan needs more candidate themes than the 5-item "fixes" list, real
    # example quotes for each, and the Digital & Tech items vs the company average.
    data["fix_candidates"] = [(t, int(tvol[t]), float(tsent[t])) for t in tvol.index
                              if tvol[t] >= 5 and tsent.get(t, 0) >= 0.45][:10]
    cq_ids, cq_txt = set(), set()
    data["cand_quotes"] = [
        {"theme": t, "text": str(r["comment"]), "role": str(r["role"]), "sentiment": r["eff_sentiment"]}
        for t, _, _ in data["fix_candidates"]
        for r in select_quotes(real, t, "Themes", "representative", 2, cq_ids, cq_txt)]
    _tech = ss[ss["mean_importance"].isna()] if len(ss) else pd.DataFrame()
    _org_t = (scores.set_index("item")["mean_support"]
              if not scores.empty and "item" in scores.columns else pd.Series(dtype=float))
    data["tech_items"] = [(r["item"], float(r["mean_support"]), float(_org_t.get(r["item"], np.nan)))
                          for _, r in _tech.iterrows()]
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

def build_site_report(ctx, site, data=None):
    """Self-contained HTML one-pager for one site. `data` = precomputed site_report_data."""
    meta, completers_only = ctx.meta, ctx.completers_only
    d = data if data is not None else site_report_data(ctx, site)
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
    def _status_color(r):
        """Status word + color: importance dims use RYG; agreement-only (Tech) use the band.
        Item rows use the stored status; dimension-summary rows (no stored status) fall
        back to the threshold rule on their mean gap."""
        if pd.isna(r["gap5"]):
            lab, hx = tech_status(r["mean_support"])
            return lab, hx
        st_ = item_status(r) if "ryg" in r.index else ryg_of(r["gap5"])
        return STATUS_WORD.get(st_, ""), RYG_HEX.get(st_, "#333")
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
    str_rows = "".join(f"<li><b>{_html.escape(t)}</b> — {v} comments, {ps*100:.0f}% positive</li>"
                       for t, v, ps in d["strengths"])
    str_rows += "".join(f"<li><b>{_html.escape(si['item'])}</b> — survey rating: support "
                        f"{si['support_100']:.0f}/100 (best-supported item)</li>"
                        for si in d.get("strength_items", []))
    str_rows = str_rows or "<li>No clearly positive themes.</li>"
    bench_rows = "".join(
        f"<tr><td>{_html.escape(t)}</td><td>{sr*100:.0f}%</td><td>{orr*100:.0f}%</td></tr>"
        for t, sr, orr in d["bench"])
    score_rows = "".join(
        f"<tr><td>{_html.escape(lab)}</td><td>{sv:.2f}</td><td>{ov:.2f}</td>"
        f"<td style='color:{'#D1495B' if sv<ov-0.05 else ('#2E9E6B' if sv>ov+0.05 else '#888')}'>"
        f"{sv-ov:+.2f}</td></tr>" for lab, sv, ov in d["scores"])

    def quote_html(q):
        n_neg = q.get("n_negative_in_theme", 0)
        tag = (f"one of {n_neg:,} negative comments tagged <b>{_html.escape(q['theme'])}</b>"
               if n_neg > 1 else f"<b>{_html.escape(q['theme'])}</b>")
        return (f"<blockquote>{_html.escape(q['text'])}"
                f"<cite>— {_html.escape(q['role'])}, {_html.escape(q['tenure'])} · {tag}</cite></blockquote>")
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
{action_plan_html}
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

def build_site_pdf(ctx, site, data=None):
    """PDF export = an exact render of the HTML site report (identical content and layout)."""
    return html_to_pdf(build_site_report(ctx, site, data))

def build_site_pptx(ctx, sites, get_data=None):
    """One slide per site. `get_data(site)` may supply cached site_report_data."""
    get_data = get_data or (lambda s: site_report_data(ctx, s))
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    prs = Presentation()
    prs.slide_width = Inches(13.33); prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    for site in sites:
        d = get_data(site)
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
        # Alphabetical, single neutral color, no rank labels — avoid implying a good-to-bad ranking.
        order = sorted(board, key=lambda x: x["site"], reverse=True)  # reverse -> A at top (barh idx0 = bottom)
        fig, ax = plt.subplots(figsize=(7.4, 3.7))
        ax.barh(range(len(order)), [b["index"] for b in order], color="#4a5a86", height=0.66)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([b["site"] for b in order], fontsize=8)
        mx = max(b["index"] for b in board)
        for i, b in enumerate(order):
            ax.text(b["index"] + mx * 0.01, i, f"{b['index']:.0f}",
                    va="center", fontsize=8, weight="bold", color="#333")
        cx = dd.get("comp_index")
        if pd.notna(cx):
            ax.axvline(cx, color="#555", ls="--", lw=1)
            ax.text(cx, len(order) - 0.4, f" company {cx:.0f}", fontsize=7.5, color="#555")
        ax.set_xlim(0, mx * 1.2); ax.set_xlabel("Index Score (0–100)", fontsize=8.5)
        ax.set_title("Overall PulsePoint Experience Index Score by site", fontsize=11,
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

def portfolio_data(ctx):
    """Roll-up data for the 10-site portfolio report card."""
    meta, scores_dim, scores = ctx.meta, ctx.scores_dim, ctx.scores
    resp_scores, comments = ctx.resp_scores, ctx.comments
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
    stand = site_standings(ctx)
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
    tex = explode_level(real10, "Themes", exclude_generic=True)
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

# --- Question-level detail grid (30 questions × 10 sites, all four metrics/cell) ----
SITE_SHORT = {
    "CA-Woodland Hills": "WH", "FL-Miami": "MIA", "FL-Tampa": "TPA", "GA-Atlanta": "ATL",
    "IL-Chicago-Willis Tower": "CHI", "IN-Indianapolis": "IND", "NY-New York (Penn1)": "NYC",
    "OH-Mason": "MSN", "VA-Norfolk": "NOR", "VA-Richmond": "RIC",
}
def _short(site):
    return SITE_SHORT.get(site, "".join(w[:1] for w in re.split(r"[ \-]+", str(site))[:3]).upper())

# light RYG cell tints (background) keyed by status / agreement band
QG_BG = {"Under Supported": "#f7d6d2", "Target": "#dcefdc", "Over Supported": "#fbe8c9",
         "Agreement-only": "#eceef1"}
QG_TECH_BG = {"Strong": "#dcefdc", "Moderate": "#fbe8c9", "Needs attention": "#f7d6d2"}
QG_MISS = "#f3f4f6"

def _portfolio_qgrid(ctx, site_order):
    """Combined per-question × per-site table: each cell carries Importance, Support,
    Gap and Friction, shaded R/Y/G by the support-vs-importance status (Digital & Tech,
    which is agreement-only, is shaded by its agreement band)."""
    ss = ctx.scores_site if ctx.scores_site is not None else pd.DataFrame()
    if ss.empty:
        return ""
    ss = ss.copy()
    for c in ("mean_importance", "mean_support", "friction"):
        ss[c] = pd.to_numeric(ss[c], errors="coerce")
    ss["gap5"] = (ss["mean_support"] - ss["mean_importance"]).round(1)
    lut = {(r["dimension"], r["item"], r["site"]): r for _, r in ss.iterrows()}
    dims = sorted(ss["dimension"].dropna().unique(), key=lambda d: DIM_ORDER.get(d, 9))
    comp_fric = ss.groupby(["dimension", "item"])["friction"].mean()

    _key = ("<span class='qs-key'><span>Imp</span><span>Sup</span>"
            "<span>Gap</span><span>Fri</span></span>")
    heads = "".join(f"<th class='qs' title='{_html.escape(str(s))}'>"
                    f"<span class='qs-code'>{_short(s)}</span>{_key}</th>" for s in site_order)
    body = ""
    ncol = len(site_order) + 1
    for dm in dims:
        items = list(pd.unique(ss[ss["dimension"] == dm]["item"].dropna()))
        items.sort(key=lambda it: -(comp_fric.get((dm, it)) if pd.notna(comp_fric.get((dm, it), np.nan)) else -1))
        body += f"<tr class='qg-dim'><td colspan='{ncol}'>{_html.escape(str(dm))}</td></tr>"
        for it in items:
            cells = ""
            for s in site_order:
                r = lut.get((dm, it, s))
                if r is None or (pd.isna(r["mean_support"]) and pd.isna(r["friction"])):
                    cells += f"<td class='qc qc-na' style='background:{QG_MISS}'>n/a</td>"
                    continue
                imp, sup, gap, fri = r["mean_importance"], r["mean_support"], r["gap5"], r["friction"]
                if pd.isna(imp):                       # agreement-only (Digital & Tech)
                    lab, _hx = tech_status(sup); bg = QG_TECH_BG.get(lab, QG_MISS)
                else:
                    bg = QG_BG.get(item_status(r), QG_MISS)
                fr = "—" if pd.isna(fri) else f"{fri:.0f}"
                # 2×2 quadrant: Imp (top-left) · Sup (top-right) / Gap (bottom-left) · Fri (bottom-right)
                cells += (f"<td class='qc' style='background:{bg}'><div class='qcw'>"
                          f"<span class='qv tl'>{_f1(imp)}</span><span class='qv tr'>{_f1(sup)}</span>"
                          f"<span class='qv bl'>{_gp(gap)}</span><span class='qv br'>{fr}</span></div></td>")
            body += f"<tr><td class='qg-q'>{_html.escape(str(it))}</td>{cells}</tr>"

    codes = " · ".join(f"<b>{_short(s)}</b> {_html.escape(str(s).split('-', 1)[-1])}" for s in site_order)
    legend = (
        "<p class='lgd'>Each cell is a 2×2 block — top row <b>Importance</b> · <b>Support</b> (1–5 scale), "
        "bottom row <b>Gap</b> (Support − Importance) · <b>Friction</b> (0–100, higher = worse); the same layout "
        "key sits under every site code. Cell color is the support-vs-importance status: "
        "<span style='background:#f7d6d2;padding:0 4px'>under-supported</span> "
        "<span style='background:#dcefdc;padding:0 4px'>on track</span> "
        "<span style='background:#fbe8c9;padding:0 4px'>exceeds needs</span>. "
        "<b>Digital &amp; Tech</b> is agreement-only (no Importance/Gap, shown as “—”) — shaded by agreement band "
        "(green ≥4.0 / amber 3.5–3.9 / red &lt;3.5). Questions are ordered by company Friction (worst first) "
        "within each dimension.</p>"
        f"<p class='lgd'>Sites (ordered by overall Index): {codes}.</p>")
    return (
        "<section class='qgrid-sec'>"
        "<h3>Question-level detail — Importance / Support / Gap / Friction by site</h3>"
        f"{legend}"
        f"<div class='qgrid-wrap'><table class='qgrid'><tr><th class='qg-qh'>Survey question</th>{heads}</tr>{body}</table></div>"
        "</section>")

def build_portfolio_report(ctx, data=None):
    """Self-contained HTML — the 10 PulsePoints portfolio report card (exec brief)."""
    completers_flag = ctx.completers_only
    dd = data if data is not None else portfolio_data(ctx)
    ch = dd["charts"]
    img = lambda k: (f"<img src='data:image/png;base64,{ch[k]}'/>" if k in ch else "")
    cx = dd["comp_index"]; board = dd["board"]
    ivals = [b["index"] for b in board]
    rng = f"{min(ivals):.0f}–{max(ivals):.0f}" if ivals else "—"
    # Index-by-site table (alphabetical; no Rank/Tier per feedback — avoid implying good-to-bad)
    lead = ""
    for b in sorted(board, key=lambda x: x["site"]):
        s = dd["sent_by_site"].get(b["site"], np.nan)
        lead += (f"<tr><td>{_html.escape(b['site'])}</td>"
                 f"<td style='font-weight:800'>{b['index']:.0f}</td>"
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
    qgrid_html = _portfolio_qgrid(ctx, [b["site"] for b in board])
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
 .qgrid-wrap{{overflow-x:auto;margin-top:8px}}
 table.qgrid{{border-collapse:collapse;width:100%;font-size:8.5px;min-width:1000px}}
 table.qgrid th,table.qgrid td{{border:1px solid #e2e5ec;padding:2px 3px;text-align:center;vertical-align:middle}}
 table.qgrid th.qg-qh,table.qgrid td.qg-q{{text-align:left;width:24%;min-width:180px;font-size:9px;line-height:1.25;color:#2b3a42;white-space:normal}}
 table.qgrid th.qs{{font-weight:800;color:#2b3a67;background:#eef2f8;white-space:nowrap}}
 tr.qg-dim td{{background:#2b3a67;color:#fff;font-weight:700;text-align:left;font-size:9.5px;letter-spacing:.03em;text-transform:uppercase;padding:3px 6px}}
 table.qgrid td.qc{{padding:0;white-space:nowrap}}
 table.qgrid td.qc .qcw{{display:grid;grid-template-columns:1fr 1fr}}
 table.qgrid td.qc .qv{{padding:1px 2px;text-align:center;line-height:1.3;font-variant-numeric:tabular-nums}}
 table.qgrid td.qc .tl,table.qgrid td.qc .tr{{font-weight:700;color:#22303a}}
 table.qgrid td.qc .bl,table.qgrid td.qc .br{{color:#43585f}}
 table.qgrid td.qc .tl,table.qgrid td.qc .bl{{border-right:1px solid rgba(30,40,60,.11)}}
 table.qgrid td.qc .tl,table.qgrid td.qc .tr{{border-bottom:1px solid rgba(30,40,60,.11)}}
 table.qgrid td.qc-na{{color:#b9bfc9;font-style:italic}}
 table.qgrid th.qs,table.qgrid td.qc{{border-left:2.5px solid #9aa3b6}}
 table.qgrid th.qs .qs-code{{display:block;font-size:10px;font-weight:800;color:#2b3a67}}
 table.qgrid th.qs .qs-key{{display:grid;grid-template-columns:1fr 1fr;column-gap:5px;margin-top:2px;
   font-size:6px;font-weight:700;color:#9aa2b1;text-transform:uppercase;letter-spacing:.02em;line-height:1.4}}
 @page{{size:Letter;margin:12mm}}
 @page qgrid{{ size:Letter landscape;margin:10mm }}
 .qgrid-sec{{ page:qgrid;break-before:page }}
 @media print{{ body{{max-width:100%;margin:0;padding:0}} .charts img{{max-width:100%}}
   h1,h3{{break-after:avoid}} tr,.rc{{break-inside:avoid}}
   table.qgrid{{min-width:0}} .qgrid-wrap{{overflow:visible}} tr.qg-dim,table.qgrid tr{{break-inside:avoid}} }}
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
<h3>Overall Index Score by site</h3>
<table><tr><th>Site</th><th>Index</th><th>vs company</th><th>Sentiment (1–5)</th></tr>{lead}</table>
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
{qgrid_html}
<div class="foot"><b>Data source &amp; method.</b> PulsePoint survey export ·
{'completed responses only' if completers_flag else 'all responses'} · {dd['n_sites']} in-scope sites. The Index Score is
the unweighted average of the four Dimension Scores (0–100) — the same calculation as the Tableau deck. Friction &amp; RYG
per the official Tableau spec.</div>
</body></html>"""

def build_portfolio_pdf(ctx, data=None):
    """PDF export = an exact render of the HTML portfolio report (identical content and layout)."""
    return html_to_pdf(build_portfolio_report(ctx, data))

def verbatim_export(ctx, site=None):
    """Long verbatim table grouped by theme for CSV/Excel export."""
    comments = ctx.comments
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

def build_excel_workbook(ctx):
    """Multi-sheet analysis workbook."""
    comments, scores, scores_dim = ctx.comments, ctx.scores, ctx.scores_dim
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
        verbatim_export(ctx).to_excel(w, sheet_name="Verbatims by theme", index=False)
        if not scores.empty:
            scores.to_excel(w, sheet_name="Scores & Friction", index=False)
            scores_dim.to_excel(w, sheet_name="Dimension scores", index=False)
    return buf.getvalue()

def build_standalone_dashboard(ctx):
    """Self-contained HTML dashboard (embedded data + vanilla JS filtering)."""
    comments, completers_flag = ctx.comments, ctx.completers_only
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