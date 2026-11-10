"""
Refresh the numbers inside the STORED Operational Action Plans (Nov 2026 review,
phase 8) without regenerating them.

    python tools/refresh_plan_evidence.py            # dry run: print old -> new
    python tools/refresh_plan_evidence.py --apply    # write the plans + change log

What it does, per plan (completed responses — the population the plans were built on):
  * keeps the priorities, their names and their order; never touches quotes
    ("whatItLooksLike") or actions;
  * rewrites comment-theme figures to the corrected data, led by people
    ("N people (P% of respondents), M comments, X% negative");
  * updates the site's own category-priority positions and scores;
  * turns "cited in X% of comments vs Y% company" into people shares against the
    company (10 in-scope sites pooled);
  * replaces every "X% positive" protect line (all were 100 − % negative) with the
    true positive and negative shares;
  * removes rank wording the standing does not support ("rank 10 of 10", "#1 overall",
    "top/bottom third", "N of 10") — standing is Above / In line with / Below;
  * checks rating-layer figures (Friction, % under-supported, agreement) against the
    scores files — those must not move;
  * records _meta.evidence_refreshed_at and _meta.evidence_flags (priorities or protect
    items the corrected data no longer clearly supports). Flags are shown on the card.

Writes docs/plan-evidence-refresh-2026-11.md (old -> new for every edited line). The
plans contain no comment text in the lines this tool edits.
"""
import argparse
import os
import re
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import action_plan  # noqa: E402
import reports as R  # noqa: E402

ALIASES = {
    "Hardware": "Hardware (Monitors / Peripherals)",
    "Connectivity": "Connectivity (WiFi / Login / Power)",
    "Focus/Concentration": "Focus / Concentration",
    "Desk Availability": "Desk Availability & Reservations",
}
THEME_NAMES = sorted(list(R.THEMES_ALL) + list(ALIASES), key=len, reverse=True)
NAME_RX = "|".join(re.escape(n) for n in THEME_NAMES)

# theme statistic clauses as the v3 plans wrote them
STAT_RX = re.compile(
    rf"(?P<name>{NAME_RX}):? (?P<n>\d[\d,]*)(?: comments)?"
    rf"(?:, (?P<neg1>\d+)% negative| \((?P<neg2>\d+)% neg(?:ative)?\)| \((?P<neg3>\d+)%\))"
    rf"(?: \((?P<k>\d+) negative comments\))?")
CAT_RX = re.compile(r"(?P<cat>Focused / Private Work|Physical Environment|Technology|Amenities & Services|"
                    r"Workplace Operations|Collaboration & Meetings|Policy / Experience)"
                    r" (?:is the #(?P<r1>\d+) priority category|ranks #(?P<r2>\d+) by priority score)"
                    r" \((?:priority score )?(?P<s>\d+)/100(?P<tail>[^)]*)\)")
CATSCORE_RX = re.compile(r"(?P<cat>Technology|Physical Environment|Focused / Private Work) priority score (?P<s>\d+)/100")
CITED_RX = re.compile(r"(?:cited (?:at|in) )(?P<a>\d+)% of comments vs\.? (?P<b>\d+)% company(?:-wide)?")
POS_RX = re.compile(r"(?P<pos>\d+)% positive(?: across (?P<n>\d+) comments)?")


def site_slug_map(ctx):
    return {action_plan._slug(s): s for s in ctx.meta["ten_sites"]}


def site_facts(ctx, site):
    d = R.site_report_data(ctx, site)
    TS = d["theme_stats"]
    P = R.build_priority(ctx.comments[ctx.comments["site"] == site], "Categories", R.theme_friction_map(ctx, site))
    P = P.reset_index(drop=True)
    comp = R.company_comments(ctx)
    n_comp = len(R.company_respondents(ctx))
    cex = R.explode_level(comp, "Themes", exclude_generic=True)
    comp_people = cex.groupby("key")["respondent"].nunique() / n_comp * 100
    return {"d": d, "TS": TS, "P": P, "comp_people_pct": comp_people}


def theme_clause(name_written, f):
    th = ALIASES.get(name_written, name_written)
    TS = f["TS"]
    if th not in TS.index:
        return f"{name_written}: no comments at this site after the November 2026 re-tag", th
    r = TS.loc[th]
    return (f"{name_written}: {int(r['people'])} people ({r['pct_resp']:.0f}% of respondents), "
            f"{int(r['comments'])} comments, {r['neg'] * 100:.0f}% negative"), th


def people_rank(f, theme, exclude=()):
    order = [t for t in f["TS"].sort_values("people", ascending=False).index if t not in exclude]
    return order.index(theme) + 1 if theme in order else None


def ordinal(n):
    return "%d%s" % (n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])


def refresh_line(text, f, plan_themes_before=()):
    """Return (new_text, notes) for one evidence / data-point line."""
    notes = []
    new = text

    def _stat(m):
        clause, th = theme_clause(m.group("name"), f)
        notes.append(("theme", th, int(m.group("n").replace(",", ""))))
        return clause
    new = STAT_RX.sub(_stat, new)

    def _cat(m):
        P = f["P"]
        cat = m.group("cat")
        if cat not in set(P["key"]):
            return f"{cat} is not among this site's ranked categories after the re-tag"
        row = P[P["key"] == cat].iloc[0]
        rank = int(P.index[P["key"] == cat][0]) + 1
        return f"{cat} is the #{rank} priority category ({row['Priority']:.0f}/100)"
    new = CAT_RX.sub(_cat, new)

    def _catscore(m):
        P = f["P"]
        row = P[P["key"] == m.group("cat")]
        return f"{m.group('cat')} priority score {row['Priority'].iloc[0]:.0f}/100" if len(row) else m.group(0)
    new = CATSCORE_RX.sub(_catscore, new)

    if CITED_RX.search(new):
        # which theme is this share about? the first theme named in the line
        names = [ALIASES.get(n, n) for n in re.findall(NAME_RX, text)]
        th = names[0] if names else None
        if th and th in f["TS"].index:
            site_pct = f["TS"].at[th, "pct_resp"]
            comp_pct = f["comp_people_pct"].get(th, np.nan)
            new = CITED_RX.sub(f"raised by {site_pct:.0f}% of respondents vs {comp_pct:.0f}% company-wide", new)
    return new, notes


def check_rating_figures(text, site, ctx):
    """Rating-layer numbers in a line must match the scores files (they cannot move)."""
    problems = []
    ss = ctx.scores_site[ctx.scores_site["site"] == site]
    for m in re.finditer(r"friction (\d+)/100 \((\d+)% under-supported\)", text):
        fr, pct = int(m.group(1)), int(m.group(2))
        hit = ss[(ss["friction"].round(0) == fr) & (ss["friction_pct"].round(0) == pct)]
        if not len(hit):
            problems.append(f"friction {fr}/100 ({pct}%) not found among this site's items")
    for m in re.finditer(r"agreement: (\d\.\d\d) vs (\d\.\d\d) company average", text):
        s, c = float(m.group(1)), float(m.group(2))
        tech = ss[ss["mean_importance"].isna()]
        if not any(abs(v - s) < 0.006 for v in tech["mean_support"]):
            problems.append(f"agreement {s} not found among this site's Digital & Tech items")
        comp = ctx.scores[ctx.scores["mean_importance"].isna()]["mean_support"]
        if not any(abs(v - c) < 0.006 for v in comp):
            problems.append(f"company agreement {c} not found")
    return problems


def refresh_protect(x, f):
    area = x.get("area", "")
    ev = x.get("evidence", "") or ""
    th = ALIASES.get(area, area)
    if not POS_RX.search(ev) or th not in R.THEMES_ALL:
        return ev, None
    TS = f["TS"]
    if th not in TS.index:
        return ev, f"protect item '{area}' has no comments after the re-tag"
    r = TS.loc[th]
    m = POS_RX.search(ev)
    rest = ev[m.end():]
    true_txt = (f"{r['pos'] * 100:.0f}% positive and {r['neg'] * 100:.0f}% negative across "
                f"{int(r['comments'])} comments from {int(r['people'])} people")
    new = ev[:m.start()] + true_txt + rest
    flag = None
    if r["pos"] < R.STRENGTH_MIN_POS_SHARE or r["pos"] <= r["neg"]:
        flag = (f"Protect item '{area}': the true positive share is {r['pos'] * 100:.0f}% "
                f"({r['neg'] * 100:.0f}% negative) — it no longer clears the What's-working bar")
    return new, flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    ctx = R.load_context(True)
    slugs = site_slug_map(ctx)
    from plan_overrides_2026_11 import OVERRIDES, PRIORITY_FLAGS  # noqa: E402  (tools/ on sys.path)
    log, problems = [], []
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for slug, site in sorted(slugs.items()):
        plan = action_plan.load_plan(site)
        if not plan:
            continue
        f = site_facts(ctx, site)
        vals = override_values(ctx, site, f)
        edits = []
        flags = []

        def edit(where, old, new):
            if new != old:
                edits.append((where, old, new))
            return new

        if "indexTier" in plan:
            std = vals["index_standing_text"]
            edits.append(("indexTier -> indexStanding", str(plan["indexTier"]), std))
            plan.pop("indexTier")
            plan["indexStanding"] = std
        for i, pr in enumerate(plan.get("priorities", []), 1):
            for e in pr.get("evidence", []):
                old = e.get("text", "")
                new = OVERRIDES.get((slug, old), None)
                new = new.format(**vals) if new is not None else refresh_line(old, f)[0]
                problems += [f"{site} P{i}: {p}" for p in check_rating_figures(new, site, ctx)]
                e["text"] = edit(f"P{i} evidence", old, new)
        for x in plan.get("evidenceConfidence", []):
            x["supportingDataPoints"] = [
                edit(f"data point — {x.get('priorityName', '')[:40]}", s,
                     (OVERRIDES[(slug, s)].format(**vals) if (slug, s) in OVERRIDES else refresh_line(s, f)[0]))
                for s in x.get("supportingDataPoints", [])]
        for x in plan.get("protectWhatIsWorking", []):
            old = x.get("evidence", "")
            if (slug, old) in OVERRIDES:
                new, flag = OVERRIDES[(slug, old)].format(**vals), None
            else:
                new, flag = refresh_protect(x, f)
            x["evidence"] = edit(f"protect — {x.get('area')}", old, new)
            if flag:
                flags.append(flag)
        plan["assumptions"] = [edit("assumption", s, OVERRIDES[(slug, s)].format(**vals) if (slug, s) in OVERRIDES else s)
                               for s in plan.get("assumptions", [])]
        flags = PRIORITY_FLAGS.get(slug, []) + flags
        meta = plan.setdefault("_meta", {})
        meta["evidence_refreshed_at"] = stamp
        meta["evidence_refresh_note"] = ("Numbers in evidence, data points and protect lines refreshed to the "
                                         "November 2026 corrected data (people-based counts, true positive "
                                         "shares, standing instead of rank). Priorities, order, quotes and "
                                         "actions unchanged; not regenerated.")
        meta["evidence_flags"] = flags
        log.append((site, edits, flags))
        if a.apply:
            action_plan.save_plan(site, plan, allow_unvalidated=True)
    out = ["# Stored action plans — evidence refresh (November 2026)", "",
           "Generated by `tools/refresh_plan_evidence.py`. Completed responses. Priorities, their order, quotes and "
           "actions are unchanged; only the numbers and wording below were edited. No comment text appears here.", ""]
    for site, edits, flags in log:
        out += [f"## {site}", ""]
        for where, old, new in edits:
            out += [f"- **{where}**", f"  - old: {old}", f"  - new: {new}"]
        if flags:
            out += ["", "**Flags (left in place, for the owner to decide):**"] + [f"- {x}" for x in flags]
        out.append("")
    if problems:
        out += ["## Rating figures that did not verify", ""] + [f"- {p}" for p in problems]
    text = "\n".join(out)
    if a.apply:
        os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
        with open(os.path.join(ROOT, "docs", "plan-evidence-refresh-2026-11.md"), "w") as fh:
            fh.write(text)
    print(text)


def override_values(ctx, site, f):
    """Numbers the hand-written overrides may use."""
    d = f["d"]
    cc = d.get("card") or {}
    areas = {a["dim"]: a for a in cc.get("areas", [])}
    v = {"index": cc.get("overall", np.nan), "index_margin": max(round(cc.get("margin", 1)), 1),
         "company_index": cc.get("org_overall", np.nan),
         "index_standing_text": R.standing_text(cc.get("standing")),
         "n_comments": d["n"], "neg_pct": d["neg"]}
    for dm, key in [("Work Activity", "wa"), ("Physical Feature", "pf"), ("Service Level", "sl"), ("Digital & Tech", "dt")]:
        a = areas.get(dm, {})
        v[f"{key}"] = a.get("score", np.nan)
        v[f"{key}_m"] = max(round(a.get("margin", 1)), 1) if pd.notna(a.get("margin")) else 1
        v[f"{key}_co"] = a.get("org", np.nan)
        v[f"{key}_std"] = R.standing_text(a.get("standing"))
    TS = f["TS"]
    for th in TS.index:
        k = re.sub(r"[^a-z]", "", th.lower())[:10]
        v[f"{k}_people"] = int(TS.at[th, "people"]); v[f"{k}_pct"] = TS.at[th, "pct_resp"]
        v[f"{k}_comments"] = int(TS.at[th, "comments"]); v[f"{k}_neg"] = TS.at[th, "neg"] * 100
        v[f"{k}_pos"] = TS.at[th, "pos"] * 100
    v["n_commenters"] = d.get("n_commenters", 0)
    return v


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
