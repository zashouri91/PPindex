"""Combined portfolio report card — round 1 and round 2 side by side.

The two rounds surveyed different sites, so a difference between rounds describes two
groups of sites (and possibly two points in time), not change at any site. Every
between-round difference is printed with its 95% margin and called a difference only
when that interval excludes zero. Each site keeps the standing from its own round.

    import reports as R, combined_report as C
    html = C.build_combined_portfolio(R.load_context(True, "round1"), R.load_context(True, "round2"))
"""
import html as _html
import os

import numpy as np
import pandas as pd
from scipy import stats

import design
import reports as R

DIMS = ["Work Activity", "Physical Feature", "Service Level", "Digital & Tech"]
SENT_ITEMS = [("score_work", "Work Activity (Q8)"), ("score_space", "Physical space (Q12)"),
              ("score_services", "Services (Q16)"), ("score_tech", "Technology (Q19)"),
              ("composite", "Overall (average of the four)")]
Z = 1.959964


def _diff_label(lo, hi):
    if any(pd.isna(v) for v in (lo, hi)):
        return "—"
    if lo > 0:
        return "Round 2 higher"
    if hi < 0:
        return "Round 2 lower"
    return "No clear difference"


def _pm(d, h, nd):
    return f"{d:+.{nd}f}" if pd.isna(h) else f"{d:+.{nd}f} ± {h:.{nd}f}"


BOOT_N, BOOT_SEED = 2000, 20261110


def _item_support(ctx):
    df = pd.read_csv(os.path.join(ctx.data_dir, "item_support.csv"))
    if ctx.completers_only:
        df = df[df["reached_end"] == 1]
    return df[df["site"].isin(ctx.meta.get("ten_sites", []))].reset_index(drop=True)


def matched_items(c1, c2):
    """Rating items answered in BOTH rounds (round 2 did not ask two Service items)."""
    a, b = _item_support(c1), _item_support(c2)
    cols = [c for c in a.columns if "|" in c]
    return [c for c in cols if c in b.columns and a[c].notna().any() and b[c].notna().any()]


def matched_round_scores(ctx, items):
    """Round average (pooled like scores_dimension.csv) on the given items, with a 95%
    respondent bootstrap stratified by site — {measure: {est, margin, se}}."""
    df = _item_support(ctx)
    S = df[items].to_numpy(float)
    dims = np.array([c.split("|", 1)[0] for c in items])
    rng = np.random.default_rng(BOOT_SEED)
    W = np.zeros((BOOT_N, len(df)))
    for _site, idx in df.groupby("site").indices.items():
        W[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), size=BOOT_N)
    ok = ~np.isnan(S)
    def scores(Wm):
        sums, cnts = Wm @ np.where(ok, S, 0.0), Wm @ ok.astype(float)
        out = {d: sums[:, dims == d].sum(1) / cnts[:, dims == d].sum(1) for d in DIMS}
        out["Index"] = np.mean(np.column_stack([out[d] for d in DIMS]), axis=1)
        return out
    pt, bt = scores(np.ones((1, len(df)))), scores(W)
    res = {}
    for m in ["Index"] + DIMS:
        lo, hi = np.percentile(bt[m], [2.5, 97.5])
        res[m] = {"est": float(pt[m][0]), "margin": float(hi - lo) / 2, "se": float(np.std(bt[m], ddof=1))}
    return res


def combined_data(c1, c2):
    out = {"rounds": []}
    for ctx in (c1, c2):
        rs = R.company_respondents(ctx)
        cm = R.company_comments(ctx)
        out["rounds"].append({"label": R.wave_info(ctx)["label"], "n_sites": len(ctx.meta.get("ten_sites", [])),
                              "n_resp": int(len(rs)), "n_comments": int(len(cm)),
                              "n_commenters": int(cm["respondent"].nunique()) if len(cm) else 0})
    # 1. Index + dimensions, round vs round, on the items BOTH rounds asked (bootstrap
    #    SEs, independent samples). Official round values are kept alongside.
    items = matched_items(c1, c2)
    all_items = [c for c in _item_support(c1).columns if "|" in c]
    out["dropped_items"] = [c.split("|", 1) for c in all_items if c not in items]
    m1, m2 = matched_round_scores(c1, items), matched_round_scores(c2, items)
    out["official"] = {m: (R.company_value(c1, m), R.company_value(c2, m)) for m in ["Index"] + DIMS}
    rows = []
    for m in ["Index"] + DIMS:
        a, b = m1[m], m2[m]
        d = b["est"] - a["est"]
        se = np.sqrt(a["se"] ** 2 + b["se"] ** 2) if pd.notna(a["se"]) and pd.notna(b["se"]) else np.nan
        h = Z * se if pd.notna(se) else np.nan
        rows.append({"measure": m, "r1": a, "r2": b, "diff": d, "margin": h,
                     "label": _diff_label(d - h, d + h) if pd.notna(h) else "—"})
    out["scores"] = rows
    # 2. Associate sentiment (1–5), Welch interval
    rs1, rs2 = R.company_respondents(c1), R.company_respondents(c2)
    srows = []
    for col, lab in SENT_ITEMS:
        x = pd.to_numeric(rs1.get(col), errors="coerce").dropna()
        y = pd.to_numeric(rs2.get(col), errors="coerce").dropna()
        if len(x) < 2 or len(y) < 2:
            continue
        vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
        se = np.sqrt(vx + vy)
        dof = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
        d = float(y.mean() - x.mean()); h = float(stats.t.ppf(0.975, dof) * se)
        srows.append({"label": lab, "r1": float(x.mean()), "r2": float(y.mean()), "diff": d, "margin": h,
                      "verdict": _diff_label(d - h, d + h)})
    out["sentiment"] = srows
    # 3. Every site, each with its own round's standing
    sites = []
    for ctx in (c1, c2):
        W = R.wave_info(ctx)
        st = R.site_standings(ctx)
        rs = R.company_respondents(ctx)
        n_by = rs.groupby("site").size().to_dict() if len(rs) else {}
        sent = rs.groupby("site")["composite"].mean().to_dict() if len(rs) else {}
        for s, v in st["overall"].items():
            x = R.site_standing(ctx, s, "Index")
            sites.append({"site": s, "round": W["label"], "key": W["key"], "n": int(n_by.get(s, 0)),
                          "index": v, "margin": x["margin"], "standing": x["standing"],
                          "dims": {d: st["dims"].get(d, {}).get(s, np.nan) for d in DIMS},
                          "sent": sent.get(s, np.nan)})
    out["sites"] = sorted(sites, key=lambda r: r["site"])
    # 4. Items: Friction + status per round; gap (Support − Importance) difference with margin
    s1 = c1.scores.set_index(["dimension", "item"]); s2 = c2.scores.set_index(["dimension", "item"])
    items = []
    for key in s1.index:
        if key not in s2.index:
            continue
        a, b = s1.loc[key], s2.loc[key]
        tech = key[0] == "Digital & Tech"
        if tech:   # agreement-only: compare mean agreement
            va, vb = float(a["mean_support"]), float(b["mean_support"])
            sa, sb = float(a["support_se"]), float(b["support_se"])
        else:
            va, vb = float(a["gap"]), float(b["gap"])
            sa, sb = float(a["gap_se"]), float(b["gap_se"])
        d = vb - va; h = Z * np.sqrt(sa ** 2 + sb ** 2)
        items.append({"dimension": key[0], "item": key[1], "tech": tech,
                      "r1": a, "r2": b, "va": va, "vb": vb, "diff": d, "margin": h,
                      "verdict": _diff_label(d - h, d + h)})
    items.sort(key=lambda r: (R.DIM_ORDER.get(r["dimension"], 9),
                              -(float(r["r2"]["friction"]) if not r["tech"] else -float(r["r2"]["mean_support"]))))
    out["items"] = items
    # 5. Themes: share of PEOPLE raising each theme, round vs round (two-proportion z, BH)
    th = []
    for ctx in (c1, c2):
        real = R.company_comments(ctx)
        th.append((R.theme_stats(real, len(R.company_respondents(ctx))), len(R.company_respondents(ctx))))
    (t1, n1), (t2, n2) = th
    keys = list(dict.fromkeys(list(t1.index[:8]) + list(t2.index[:8])))
    trows = []
    for k in keys:
        x1 = int(t1.at[k, "people"]) if k in t1.index else 0
        x2 = int(t2.at[k, "people"]) if k in t2.index else 0
        p1, p2 = x1 / n1, x2 / n2
        pool = (x1 + x2) / (n1 + n2)
        se = np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
        p = float(2 * (1 - stats.norm.cdf(abs((p2 - p1) / se)))) if se > 0 else 1.0
        trows.append({"theme": k, "p1": p1 * 100, "p2": p2 * 100, "x1": x1, "x2": x2, "p": p})
    if trows:
        df = pd.DataFrame(trows).sort_values("p").reset_index(drop=True)
        m = len(df)
        df["q"] = (df["p"] * m / (df.index + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
        df["differs"] = df["q"] < 0.05
        df = df.sort_values("p2", ascending=False)
        trows = df.to_dict("records")
    out["themes"] = trows
    return out


COMBINED_CSS = """
 .hdr2{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:6px 0 10px}
 .hdr2 > div{border:1px solid var(--rule);border-radius:10px;padding:10px 14px;background:var(--panel)}
 .hdr2 .t{font-size:14px;font-weight:700;color:var(--navy)} .hdr2 .s{font-size:13px;color:var(--ink-2)}
 .hdr2 .big{font-size:28px;font-weight:800;color:var(--navy)}
 td.hm{font-weight:700} .rtag{font-size:11px;font-weight:700;padding:1px 6px;border-radius:8px;border:1px solid var(--rule);
   white-space:nowrap}
 table.allsites{font-size:12px} table.allsites td.sn{min-width:170px} table.allsites .lown{display:block}
 .lown{font-size:11px;color:var(--ink-2);font-weight:400}
 .callout{border-left:4px solid var(--navy);background:var(--panel);padding:8px 12px;margin:8px 0;font-size:13px}
"""


def build_combined_portfolio(c1, c2, data=None):
    esc = _html.escape
    dd = data if data is not None else combined_data(c1, c2)
    pop = R.population_label(c1.completers_only)
    low_n = R.wave_info(c2).get("low_n", 30)
    r1, r2 = dd["rounds"]
    def official(ctx):
        iv = ctx.intervals
        row = iv[(iv["site"] == R.wave_info(ctx)["company_row"]) & (iv["measure"] == "Index")]
        return {"est": R.company_value(ctx, "Index"), "margin": float(row["margin"].iloc[0]) if len(row) else np.nan}
    def head(r, s):
        return (f"<div><div class='s'>{esc(r['label'])} · {r['n_sites']} sites</div>"
                f"<div class='big'>{R.fmt_margin(s['est'], s['margin'], 1)}</div>"
                f"<div class='s'>Overall PulsePoint Experience Index (0–100)</div>"
                f"<div class='s'>{r['n_resp']:,} respondents · {r['n_comments']:,} comments from "
                f"{r['n_commenters']:,} people</div></div>")
    header = f"<div class='hdr2 keep'>{head(r1, official(c1))}{head(r2, official(c2))}</div>"
    callout = ("<div class='callout'><b>Read this first.</b> The two rounds surveyed <b>different sites</b>. "
               "A difference between rounds describes two groups of sites (and possibly two points in time) — "
               "it is not a change at any site. Each site's standing is against its own round's average.</div>")
    sc_rows = "".join(
        f"<tr><td>{'<b>Index</b>' if r['measure'] == 'Index' else esc(r['measure'])}</td>"
        f"<td class='num'>{R.fmt_margin(r['r1']['est'], r['r1']['margin'], 1)}</td>"
        f"<td class='num'>{R.fmt_margin(r['r2']['est'], r['r2']['margin'], 1)}</td>"
        f"<td class='num'>{_pm(r['diff'], r['margin'], 1)}</td>"
        f"<td>{esc(r['label'])}</td></tr>" for r in dd["scores"])
    se_rows = "".join(
        f"<tr><td>{esc(r['label'])}</td><td class='num'>{r['r1']:.2f}</td><td class='num'>{r['r2']:.2f}</td>"
        f"<td class='num'>{r['diff']:+.2f} ± {r['margin']:.2f}</td><td>{esc(r['verdict'])}</td></tr>"
        for r in dd["sentiment"])
    # every site
    allv = [v for s in dd["sites"] for v in s["dims"].values() if pd.notna(v)]
    vmin, vmax = (min(allv), max(allv)) if allv else (0, 100)
    site_rows = ""
    for s in dd["sites"]:
        lown = (f" <span class='lown'>(small sample, n={s['n']})</span>"
                if s["key"] != "round1" and s["n"] < low_n else "")
        cells = "".join(R._seq_cell(v, vmin, vmax, f"{v:.0f}" if pd.notna(v) else "—") for v in s["dims"].values())
        site_rows += (f"<tr><td class='sn'>{esc(s['site'])}{lown}</td><td><span class='rtag'>{esc(s['round'])}</span></td>"
                      f"<td class='num'><b>{R.fmt_margin(s['index'], s['margin'])}</b></td>"
                      f"<td class='stand'>{esc(R.STANDING_SHORT.get(s['standing'], ''))}</td>{cells}"
                      f"<td class='num'>{R._sent_txt(s['sent'])}</td></tr>")
    # items
    it_rows, cur = "", None
    for r in dd["items"]:
        if r["dimension"] != cur:
            cur = r["dimension"]
            unit = "agreement (1–5)" if r["tech"] else "Friction (0–100) · status"
            it_rows += f"<tr class='qg-dim'><td colspan='5'>{esc(cur)} — {unit}</td></tr>"
        if r["tech"]:
            a = f"◇ {r['va']:.2f}"; b = f"◇ {r['vb']:.2f}"
            dtxt = f"{r['diff']:+.2f} ± {r['margin']:.2f}"
        else:
            def cell(x):
                stt = R.item_status(x)
                return (f"<span class='st {design.STATUS_CLASS.get(stt, 'st-na')}'>"
                        f"{design.STATUS_ICON.get(stt, '')} {float(x['friction']):.0f}</span>")
            a, b = cell(r["r1"]), cell(r["r2"])
            dtxt = f"{r['diff']:+.2f} ± {r['margin']:.2f}"
        it_rows += (f"<tr><td>{esc(str(r['item']))}</td><td class='num'>{a}</td><td class='num'>{b}</td>"
                    f"<td class='num'>{dtxt}</td><td>{esc(r['verdict'])}</td></tr>")
    th_rows = "".join(
        f"<tr><td>{esc(t['theme'])}</td><td class='num'>{t['x1']:,} ({t['p1']:.0f}%)</td>"
        f"<td class='num'>{t['x2']:,} ({t['p2']:.0f}%)</td><td class='num'>{t['p2'] - t['p1']:+.0f} pts</td>"
        f"<td>{'Differs' if t['differs'] else 'No clear difference'}</td></tr>" for t in dd["themes"])
    dropped = dd.get("dropped_items") or []
    if dropped:
        names = "; ".join(f"“{esc(i)}” ({esc(d)})" for d, i in dropped)
        o = dd["official"]
        dropped_note = (f"<b>Like for like:</b> Round 2 did not ask {names}, so this table compares both rounds on the "
                        f"items both asked. Official values with every item: Round 1 Index {o['Index'][0]:.1f}, "
                        f"Service Level {o['Service Level'][0]:.1f}; Round 2 Index {o['Index'][1]:.1f}, Service Level "
                        f"{o['Service Level'][1]:.1f}. ")
    else:
        dropped_note = ""
    css = design.stylesheet(R.PORTFOLIO_CSS + COMBINED_CSS)
    dim_heads = "".join(f"<th class='num'>{esc(d)}</th>" for d in DIMS)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>PulsePoints — Combined Portfolio Report Card (Round 1 + Round 2)</title>
<style>{css}</style></head><body>
<h1>PulsePoints — Combined Portfolio Report Card</h1>
<div class="sub">Round 1 ({r1['n_sites']} sites) and Round 2 ({r2['n_sites']} sites) · {esc(pop)}</div>
{header}
{callout}
<h2>Round averages</h2>
<table><thead><tr><th>Measure</th><th class='num'>Round 1 ± margin</th><th class='num'>Round 2 ± margin</th>
<th class='num'>Round 2 − Round 1</th><th>Reading</th></tr></thead>{sc_rows}</table>
<p class="note">Each round's average pools its own sites (Round 1: the 10 in-scope sites; Round 2: all {r2['n_sites']} sites).
{dropped_note}± is the 95% margin from resampling respondents within each site; the difference's margin combines both
(independent samples). “Round 2 higher / lower” only when the difference's interval excludes zero.</p>
<h2>Associate sentiment (1–5)</h2>
<table><thead><tr><th>Summary question</th><th class='num'>Round 1</th><th class='num'>Round 2</th>
<th class='num'>Difference ± margin</th><th>Reading</th></tr></thead>{se_rows}</table>
<p class="note">Welch 95% interval on the difference of means. Q19's header text differs between the two exports;
it is treated as the same question (owner's decision).</p>
<h2>Every site</h2>
<table class='allsites'><thead><tr><th>Site</th><th>Round</th><th class='num'>Index ± 95% margin</th><th>Standing vs own round</th>
{dim_heads}<th class='num'>Sentiment (1–5)</th></tr></thead>{site_rows}</table>
<p class="note">Sites are alphabetical — no ranking. Standing compares a site with <b>its own round's</b> average
(Round 1: {R.company_value(c1, 'Index'):.1f}; Round 2: {R.company_value(c2, 'Index'):.1f}). Dimension cells share one
colour scale across both rounds (lighter = lower). Round 2 sites with fewer than {low_n} responses are marked
“small sample”.</p>
<h2>Survey items — Round 1 vs Round 2</h2>
<table class='qgrid'><thead><tr><th>Survey question</th><th class='num'>Round 1</th><th class='num'>Round 2</th>
<th class='num'>Difference ± margin</th><th>Reading</th></tr></thead>{it_rows}</table>
<p class="note">For items with an Importance rating the cells show Friction and the official status
(▼ Needs attention · ● On track · ▲ Exceeds needs); the difference column compares the <b>gap</b>
(Support − Importance, 1–5), where a positive value means Round 2 is better supported. Digital &amp; Tech items
(◇) compare mean agreement (1–5). 95% margins treat respondents as independent and do not allow for clustering
within sites, so they are slightly narrow.</p>
<h2>Comment themes — share of respondents raising each</h2>
<table><thead><tr><th>Theme</th><th class='num'>Round 1 people (%)</th><th class='num'>Round 2 people (%)</th>
<th class='num'>Difference</th><th>Reading</th></tr></thead>{th_rows}</table>
<p class="note">Top themes in either round. Two-proportion test per theme with a Benjamini–Hochberg correction across
the themes shown; “Differs” = adjusted p &lt; 0.05. Generic praise / dissatisfaction buckets are excluded.
The keyword taxonomy was tuned on Round 1 comments; its precision on Round 2 has not been checked.</p>
<div class="foot"><b>Data source &amp; method.</b> PulsePoint survey exports, Round 1 and Round 2 · {esc(pop)}.
Index = unweighted average of the four Dimension Scores (0–100), the Tableau calculation, applied to each round
separately. {R.company_def(c1)} {R.company_def(c2)} {R.tagging_accuracy_note()}</div>
</body></html>"""


def build_combined_pdf(c1, c2, data=None):
    return R.html_to_pdf(build_combined_portfolio(c1, c2, data))
