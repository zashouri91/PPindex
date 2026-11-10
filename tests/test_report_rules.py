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
