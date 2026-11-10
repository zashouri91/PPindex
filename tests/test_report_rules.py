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
