"""
PulsePoint Open-Ended Response Analyzer
========================================
An interactive tool to review and analyze open-ended survey comments:
  • Gap Analysis .......  why categories are under/over-supported (ratings + comments)
  • Theme & Sentiment ..  trends by theme, sentiment, site, role, tenure
  • Tagging Workspace ..  review & override auto-tags, flag junk, export

Run:  streamlit run app.py

The computation and the HTML/PDF/PPTX builders live in reports.py (importable
without Streamlit); this file holds the UI and the st.cache_data wrappers.
"""
import os, html as _html, io, zipfile
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import altair as alt

import action_plan  # Operational Action Plan — "What to Do Next" section (LLM-generated, stored per site)
import reports as R
from reports import (SENT_COLORS, RYG_HEX, SCORE_LABELS, OVERRIDES, themes_for_item,
                     explode_level, build_priority, exemplars, build_hotspots, build_driver)

st.set_page_config(page_title="PulsePoint Comment Analyzer", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Data loading (cached) — overrides are merged in by reports.load_all
# ---------------------------------------------------------------------------
load_all = st.cache_data(R.load_all)
load_scores = st.cache_data(R.load_scores)
load_extra = st.cache_data(R.load_extra)

comments_full, meta, GAPS = load_all()
SCORES = load_scores()
THEME_TO_CAT = meta["theme_to_category"]
resp_scores_full, emerging = load_extra()

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
CTX = R.make_context(comments_full, resp_scores_full, SCORES, meta, completers_only)
comments, resp_scores = CTX.comments, CTX.resp_scores
scores, scores_site, scores_dim = CTX.scores, CTX.scores_site, CTX.scores_dim
if completers_only:
    gap, gap_site = GAPS["comp"], GAPS["comp_site"]
    st.sidebar.caption(f"✓ Completed only · {n_comp:,} respondents")
else:
    gap, gap_site = GAPS["all"], GAPS["all_site"]
    st.sidebar.caption(f"Including all {meta['n_responses']:,} respondents")
THEME_FRICTION_ALL = CTX.theme_friction_all

# ---------------------------------------------------------------------------
# Context-bound wrappers around reports.py (cached where the original was)
# ---------------------------------------------------------------------------
def theme_friction_map(site=None):
    return R.theme_friction_map(CTX, site)

@st.cache_data(show_spinner=False)
def site_report_data(site, completers_flag):
    return R.site_report_data(CTX, site)

def build_site_report(site):
    return R.build_site_report(CTX, site, site_report_data(site, completers_only))

@st.cache_data(show_spinner=False)
def build_site_pdf(site, completers_flag):
    return R.build_site_pdf(CTX, site, site_report_data(site, completers_flag))

def build_site_pptx(sites, completers_flag):
    return R.build_site_pptx(CTX, sites, lambda s: site_report_data(s, completers_flag))

@st.cache_data(show_spinner=False)
def portfolio_data(completers_flag):
    return R.portfolio_data(CTX)

def build_portfolio_report(completers_flag):
    return R.build_portfolio_report(CTX, portfolio_data(completers_flag))

@st.cache_data(show_spinner=False)
def build_portfolio_pdf(completers_flag):
    return R.build_portfolio_pdf(CTX, portfolio_data(completers_flag))

def verbatim_export(site=None):
    return R.verbatim_export(CTX, site)

@st.cache_data(show_spinner=False)
def build_excel_workbook(completers_flag):
    return R.build_excel_workbook(CTX)

@st.cache_data(show_spinner=False)
def build_standalone_dashboard(completers_flag):
    return R.build_standalone_dashboard(CTX)

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
    st.caption("A company-wide exec brief across the 10 in-scope sites: Index Score by site with its margin and standing, a site×dimension "
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
        st.caption("An auto-generated, validate-before-acting **What to Do Next** section. On the site card it "
                   "renders after the theme tables and before the quotes. Grounded only in this site's evidence; "
                   "generated once and stored.")
        _prov = action_plan.configured_provider()
        _pm = action_plan.plan_meta(rsite)
        if _pm:
            st.caption(f"Current plan for **{rsite}**: generated {_pm.get('generated_at','?')} · "
                       f"{_pm.get('provider','?')}/{_pm.get('model','?')}"
                       + (" · aggregate-only" if _pm.get('include_verbatims') is False else ""))
        else:
            st.caption(f"No action plan generated yet for **{rsite}** — it appears below the Vs. org theme table once created.")
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
