"""Rules for what the site cards print (Nov 2026 review, phases 2+)."""
import pytest

import reports as R


@pytest.fixture(scope="module")
def ctx():
    return R.load_context(completers_only=True)


@pytest.fixture(scope="module")
def site_data(ctx):
    return {s: R.site_report_data(ctx, s) for s in ctx.meta["ten_sites"]}


def _theme_shares(ctx, site, theme):
    real = ctx.comments[(ctx.comments["site"] == site) & ~ctx.comments["eff_junk"]]
    g = R.explode_level(real, "Themes")
    g = g[g["key"] == theme]
    return len(g), (g["eff_sentiment"] == "Positive").mean(), (g["eff_sentiment"] == "Negative").mean()


def test_whats_working_prints_true_positive_share(ctx, site_data):
    for site, d in site_data.items():
        for theme, n, pos in d["strengths"]:
            n2, pos2, neg2 = _theme_shares(ctx, site, theme)
            assert n == n2 and pos == pytest.approx(pos2), (site, theme)
            assert n >= R.STRENGTH_MIN_COMMENTS
            assert pos >= R.STRENGTH_MIN_POS_SHARE and pos > neg2
        # fewer than two qualifying themes -> topped up from survey ratings
        assert len(d["strengths"]) + len(d["strength_items"]) >= min(2, len(d["strengths"]) + 4)


def test_woodland_hills_reception_no_longer_91(site_data):
    d = site_data["CA-Woodland Hills"]
    assert all(t != "Reception & Entry" for t, _n, _p in d["strengths"])


def _toy_comments():
    import pandas as pd
    rows = [  # invented: themes and sentiment only, no text needed
        (1, "Positive / Appreciation", "Positive"),
        (2, "Positive / Appreciation|RTO / Hybrid Policy", "Positive"),
        (3, "Overall Dissatisfaction", "Negative"),
        (4, "Temperature / Thermal", "Negative"),
        (5, "Temperature / Thermal|Lighting", "Negative"),
    ]
    return pd.DataFrame([{"comment_id": i, "eff_themes": th, "eff_sentiment": s, "eff_junk": False,
                          "respondent": i, "site": "X"} for i, th, s in rows])


def test_generic_themes_defined_once_and_used_by_action_plan():
    import action_plan
    from taxonomy import GENERIC_THEMES
    assert GENERIC_THEMES == {"Positive / Appreciation", "Overall Dissatisfaction"}
    assert action_plan.GENERIC_THEMES is GENERIC_THEMES
    assert not hasattr(action_plan, "EXCLUDED_THEMES")


def test_priorities_exclude_generic_buckets():
    df = _toy_comments()
    themes = R.build_priority(df, "Themes", {})["key"].tolist()
    assert not set(themes) & {"Positive / Appreciation", "Overall Dissatisfaction"}
    cats = R.build_priority(df, "Categories", {})
    pe = cats[cats["key"] == "Policy / Experience"]
    assert len(pe) == 1 and int(pe["volume"].iloc[0]) == 1   # only the RTO comment counts


def test_generic_comments_still_count_in_browser_views():
    df = _toy_comments()
    assert set(R.explode_level(df, "Themes")["key"]) >= {"Positive / Appreciation"}


def test_no_represents_similar_comments_claim(ctx):
    html = R.build_site_report(ctx, "IN-Indianapolis")
    assert "represents ~" not in html
    assert "similar comments" not in html


def test_item_status_comes_from_stored_column(ctx):
    import re
    html = R.build_site_report(ctx, "NY-New York (Penn1)")
    # gap +1.515 rounds to +1.5 (Target by threshold) but the official status is Over Supported
    row = re.search(r"Planned in-person meetings at a PulsePoint</td>(.*?)</tr>", html).group(1)
    assert "Exceeds needs" in row
    ss = ctx.scores_site
    imp = ss[ss["mean_importance"].notna()]
    for _, r in imp.iterrows():
        assert R.item_status(r) == r["ryg"]


def test_copy_theme_count_and_population():
    from taxonomy import THEMES
    for flag in (True, False):
        c = R.load_context(completers_only=flag)
        site_html = R.build_site_report(c, "FL-Miami")
        port_html = R.build_portfolio_report(c)
        for html in (site_html, port_html):
            assert f"{len(THEMES)} themes" in html and "28 themes" not in html
            assert R.population_label(flag) in html


def test_standing_rule():
    assert R.standing_label(70, 72, 69.0) == R.STANDING_ABOVE
    assert R.standing_label(66, 68.9, 69.0) == R.STANDING_BELOW
    assert R.standing_label(67, 69.3, 69.0) == R.STANDING_INLINE
    assert R.fmt_margin(68.17, 1.11) == "68 ± 1" and R.fmt_margin(70.0, 0.2) == "70 ± 1"


def test_index_standings_match_review(site_data):
    want = {s: R.STANDING_INLINE for s in site_data}
    want.update({"GA-Atlanta": R.STANDING_ABOVE, "VA-Richmond": R.STANDING_BELOW,
                 "CA-Woodland Hills": R.STANDING_BELOW})
    assert {s: d["card"]["standing"] for s, d in site_data.items()} == want


def test_no_rank_language_in_cards_or_payload(ctx, site_data):
    import re
    import action_plan
    for site, d in site_data.items():
        p = action_plan.build_report_card_payload(d, include_verbatims=False)
        flat = str(p)
        assert "rank_of_sites" not in flat and "tier" not in flat.lower()
    port = R.build_portfolio_report(ctx)
    assert not re.search(r"third|#\d+ of|[Rr]ank\b", port)
    order = [b["site"] for b in R.portfolio_data(ctx)["board"]]
    assert order == sorted(order)


def test_company_is_ten_sites_pooled(ctx, site_data):
    comp = R.company_comments(ctx)
    assert set(comp["site"]) == set(ctx.meta["ten_sites"])
    d = site_data["IN-Indianapolis"]
    for theme, _site_rate, org_rate in d["bench"]:
        want = comp["eff_themes"].str.contains(theme, regex=False).sum() / len(comp)
        assert org_rate == pytest.approx(want), theme
    html = R.build_site_report(ctx, "IN-Indianapolis")
    assert R.COMPANY_DEF in html and R.COMPANY_DEF in R.build_portfolio_report(ctx)


def test_sentiment_table_uses_margin_and_company(ctx, site_data):
    comp = R.company_respondents(ctx)
    for site, d in site_data.items():
        for a in d["agreement"]:
            col = R.DIM_TO_SCORECOL[a["dim"]]
            assert a["org_mean"] == pytest.approx(round(comp[col].mean(), 2))
            assert a["standing"] == R.standing_label(a["lo"], a["hi"], a["org_mean"])
    from conftest import strip_comment_text
    html = strip_comment_text(R.build_site_report(ctx, "GA-Atlanta"))
    assert "Welch" not in html and "p=" not in html and "significant" not in html
