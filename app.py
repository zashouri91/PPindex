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
    # official Friction scores (Tableau spec) — only exist for the 10 in-scope sites
    ss = scores_site[scores_site["site"] == site].dropna(subset=["friction"]) if not scores_site.empty else pd.DataFrame()
    data["under"] = ss.sort_values("friction", ascending=False).head(4) if len(ss) else pd.DataFrame()
    data["over"] = ss[ss["ryg"] == "Over Supported"].head(2) if len(ss) else pd.DataFrame()
    data["in_scope"] = len(ss) > 0
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

def _site_charts(d):
    """Return dict of base64 PNG charts for a site report."""
    charts = {}
    # donut
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    vals = [d["pos"], d["neu"], d["neg"]]
    ax.pie(vals, colors=[SENT_HEX["Positive"], SENT_HEX["Neutral / Mixed"], SENT_HEX["Negative"]],
           startangle=90, wedgeprops=dict(width=0.42))
    ax.text(0, 0, f"{d['neg']:.0f}%\nneg", ha="center", va="center", fontsize=11, weight="bold")
    ax.set(aspect="equal"); charts["donut"] = _b64png(fig)
    # top themes bar
    tt = d["top_themes"].head(7)[::-1]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.barh(range(len(tt)), tt.values, color="#4C78A8")
    ax.set_yticks(range(len(tt))); ax.set_yticklabels([t[:26] for t in tt.index], fontsize=8)
    ax.set_xlabel("mentions", fontsize=8); ax.tick_params(axis="x", labelsize=8)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    charts["themes"] = _b64png(fig)
    # friction bars (Tableau spec)
    if len(d["under"]):
        gg = d["under"].sort_values("friction")
        fig, ax = plt.subplots(figsize=(4.4, 2.8))
        ax.barh(range(len(gg)), gg["friction"], color="#D1495B")
        ax.set_yticks(range(len(gg))); ax.set_yticklabels([str(i)[:30] for i in gg["item"]], fontsize=7)
        ax.set_xlabel("Friction (0–100, higher = worse)", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        charts["gap"] = _b64png(fig)
    return charts

def build_site_report(site):
    """Self-contained HTML one-pager for one site."""
    d = site_report_data(site, completers_only)
    if d["n"] == 0:
        return f"<html><body style='font-family:sans-serif'><h2>{_html.escape(site)}</h2><p>No comments for this site.</p></body></html>"
    charts = d["charts"]
    P, tops = d["P"], d["P"]["key"].tolist()
    narrative = (f"{d['n']} associates at <b>{_html.escape(site)}</b> left a substantive comment. "
                 f"Overall sentiment is <b>{d['neg']:.0f}% negative</b> and {d['pos']:.0f}% positive. ")
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
        f"<td>{int(r['neg_share']*100)}%</td><td>{int(round(r['Priority']))}</td></tr>"
        for _, r in P.iterrows()) or "<tr><td colspan='4'>Not enough tagged comments to rank.</td></tr>"
    def _gap_row(r):
        comp = "" if pd.isna(r["support_100"]) else f"{r['support_100']:.0f}"
        fpct = "" if pd.isna(r["friction_pct"]) else f"{int(r['friction_pct'])}%"
        return (f"<tr><td>{_html.escape(str(r['item']))}</td><td>{comp}</td>"
                f"<td>{r['friction']:.0f}</td><td>{fpct}</td></tr>")
    gap_rows = "".join(_gap_row(r) for _, r in d["under"].iterrows()) or (
        "<tr><td colspan='4'>Not one of the 10 in-scope sites — no official scores. "
        "The comment analysis above still applies.</td></tr>")
    fix_rows = "".join(f"<li><b>{_html.escape(t)}</b> — {v} comments, {int(ns*100)}% negative</li>"
                       for t, v, ns in d["fixes"]) or "<li>No clearly negative themes.</li>"
    str_rows = "".join(f"<li><b>{_html.escape(t)}</b> — {v} comments, {int(ps*100)}% positive</li>"
                       for t, v, ps in d["strengths"]) or "<li>No clearly positive themes.</li>"
    bench_rows = "".join(
        f"<tr><td>{_html.escape(t)}</td><td>{sr*100:.0f}%</td><td>{orr*100:.0f}%</td>"
        f"<td>{'—' if orr==0 else f'{sr/orr:.1f}×'}</td></tr>" for t, sr, orr in d["bench"])
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
 .foot{{color:#aaa;font-size:11px;margin-top:24px}}
</style></head><body>
<h1>PulsePoint Experience — {_html.escape(site)}</h1>
<div class="sub">Open-ended comment summary · {d['n']} substantive comments</div>
<div class="narr">{narrative}</div>
<div class="charts">{img('donut')}{img('themes')}{img('gap')}</div>
<h3>Top priorities to address</h3>
<table><tr><th>Category</th><th>#&nbsp;comments</th><th>%&nbsp;negative</th><th>priority</th></tr>{prio_rows}</table>
<div class="two"><div>
<h3>⚠️ What to fix</h3><ul>{fix_rows}</ul></div><div>
<h3>✅ What's working</h3><ul>{str_rows}</ul></div></div>
<h3>Highest-Friction items (Tableau spec)</h3>
<table><tr><th>Item</th><th>composite</th><th>friction</th><th>% w/ friction</th></tr>{gap_rows}</table>
<div class="two"><div>
<h3>Vs. org — theme rates</h3>
<table><tr><th>Theme</th><th>site</th><th>org</th><th>lift</th></tr>{bench_rows}</table></div><div>
<h3>Vs. org — satisfaction (1–5)</h3>
<table><tr><th>Area</th><th>site</th><th>org</th><th>Δ</th></tr>{score_rows}</table></div></div>
<h3>In their words — what to fix</h3>{neg_quotes}
{('<h3>In their words — what’s working</h3>'+pos_quotes) if pos_quotes else ''}
<div class="foot">Generated by the PulsePoint Open-Ended Response Analyzer · {'completed responses only' if completers_only else 'all responses'} · Friction &amp; Composite per the official Tableau spec · Priority = 40% volume + 35% share negative + 25% Friction.</div>
</body></html>"""

@st.cache_data(show_spinner=False)
def build_site_pdf(site, completers_flag):
    """Native PDF one-pager via reportlab (mirrors the HTML)."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib import colors as rc
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                    Image as RLImage)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    d = site_report_data(site, completers_flag)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6*inch, bottomMargin=0.6*inch,
                            leftMargin=0.6*inch, rightMargin=0.6*inch)
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=18, spaceAfter=2)
    h3 = ParagraphStyle("h3", parent=ss["Heading3"], textColor=rc.HexColor("#2b3a67"), fontSize=11, spaceBefore=10)
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=13)
    quote = ParagraphStyle("q", parent=body, leftIndent=8, textColor=rc.HexColor("#333"),
                           borderColor=rc.HexColor("#D1495B"), fontName="Helvetica-Oblique")
    story = [Paragraph(f"PulsePoint Experience — {_html.escape(site)}", h1)]
    if d["n"] == 0:
        story.append(Paragraph("No comments for this site.", body))
        doc.build(story); return buf.getvalue()
    story.append(Paragraph(f"Open-ended comment summary · {d['n']} substantive comments", body))
    tops = d["P"]["key"].tolist()
    narr = (f"Overall sentiment is <b>{d['neg']:.0f}% negative</b> and {d['pos']:.0f}% positive. "
            f"Top area to address: <b>{tops[0] if tops else '—'}</b>.")
    story += [Spacer(1, 6), Paragraph(narr, body)]
    charts = d["charts"]
    imgs = []
    for k in ("donut", "themes", "gap"):
        if k in charts:
            imgs.append(RLImage(io.BytesIO(base64.b64decode(charts[k])),
                                width=2.1*inch, height=1.7*inch, kind="proportional"))
    if imgs:
        story += [Spacer(1, 6), Table([imgs], colWidths=[2.3*inch]*len(imgs))]

    def tbl(header, rows, widths):
        data = [header] + rows
        t = Table(data, colWidths=widths, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("TEXTCOLOR", (0, 0), (-1, 0), rc.HexColor("#555")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, rc.HexColor("#e6e6e6")),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT")]))
        return t
    story += [Paragraph("Top priorities to address", h3),
              tbl(["Category", "# comments", "% neg", "priority"],
                  [[r["key"], int(r["volume"]), f"{int(r['neg_share']*100)}%", int(round(r["Priority"]))]
                   for _, r in d["P"].iterrows()], [2.6*inch, 1.1*inch, 0.9*inch, 0.9*inch])]
    if len(d["under"]):
        story += [Paragraph("Highest-Friction items (Tableau spec)", h3),
                  tbl(["Item", "composite", "friction", "% w/ friction"],
                      [[str(r["item"])[:56],
                        "" if pd.isna(r["support_100"]) else f"{r['support_100']:.0f}",
                        f"{r['friction']:.0f}",
                        "" if pd.isna(r["friction_pct"]) else f"{int(r['friction_pct'])}%"]
                       for _, r in d["under"].iterrows()], [3.4*inch, 0.9*inch, 0.8*inch, 0.9*inch])]
    if d["quotes"]:
        story.append(Paragraph("In their words — what to fix", h3))
        for q in d["quotes"][:4]:
            rep = f" · represents ~{q['represents']} similar" if q.get("represents", 0) >= 3 else ""
            story += [Paragraph(f"“{_html.escape(q['text'])}”", quote),
                      Paragraph(f"— {q['role']}, {q['tenure']} · <b>{q['theme']}</b>{rep}",
                                ParagraphStyle("c", parent=body, fontSize=7.5, textColor=rc.HexColor("#888"))),
                      Spacer(1, 4)]
    doc.build(story)
    return buf.getvalue()

@st.cache_data(show_spinner=False)
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
RYG_COLORS = {"Under Supported": "#D1495B", "Target": "#E9C46A",
              "Over Supported": "#4C78A8", "Agreement-only": "#9AA0A6"}

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
        st.download_button("⬇️ PDF", build_site_pdf(rsite, completers_only), f"pulsepoint_{safe}.pdf",
                           "application/pdf", use_container_width=True)
        st.download_button("⬇️ PowerPoint", build_site_pptx([rsite], completers_only), f"pulsepoint_{safe}.pptx",
                           "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                           use_container_width=True)
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
