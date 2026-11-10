"""Automated checks on the rendered cards (Nov 2026 review: phases 5-7)."""
import re

import pytest

import reports as R
from conftest import strip_comment_text


@pytest.fixture(scope="module", params=[True, False], ids=["completed", "all"])
def ctx(request):
    return R.load_context(completers_only=request.param)


@pytest.fixture(scope="module")
def site_html(ctx):
    return {s: R.build_site_report(ctx, s) for s in ctx.meta["ten_sites"]}


SECTIONS = ["At a glance", "Importance vs. Support", "Associate sentiment", "Top priorities",
            "What needs attention", "Themes that differ here", "In their words", "All comments by category",
            "Respondent profile", "Data source &amp; method"]


def test_section_order(site_html):
    for site, h in site_html.items():
        pos = [h.find(s) for s in SECTIONS]
        assert all(p >= 0 for p in pos), (site, [s for s, p in zip(SECTIONS, pos) if p < 0])
        assert pos == sorted(pos), site


def test_banned_wording(ctx, site_html):
    port = R.build_portfolio_report(ctx)
    for h in list(site_html.values()) + [port]:
        own = strip_comment_text(h)
        assert not re.search(r"\bthird\b", own, re.I)
        assert not re.search(r"#\d+ of 10", own)
        assert "represents ~" not in own and "feel it" not in own


def test_images_have_alt_text(ctx, site_html):
    for h in list(site_html.values()):
        for img in re.findall(r"<img\b[^>]*>", h):
            assert re.search(r"\balt\s*=\s*['\"][^'\"]+['\"]", img), img[:80]


def test_no_tenure_beside_comments(ctx, site_html):
    tenures = [t for t in ctx.meta["tenures"]]
    for site, h in site_html.items():
        # attribution lives in <cite> (quotes) and the trailing note span in the comment list
        attributions = re.findall(r"<cite>(.*?)</cite>", h, re.S) + re.findall(r"<span class='note'>— (.*?)</span>", h)
        for a in attributions:
            assert not any(t in a for t in tenures), (site, a[:80])


def test_roles_only_for_groups_of_ten(ctx, site_html):
    roles = set(ctx.meta["roles"])
    for site, h in site_html.items():
        ok = R.attributable_roles(ctx, site)
        shown = set()
        for a in re.findall(r"<cite>(.*?)</cite>", h, re.S) + re.findall(r"<span class='note'>— (.*?)</span>", h):
            shown |= {r for r in roles if r in a}
        assert shown <= ok, (site, shown - ok)


def test_small_counts_print_as_lt5(ctx, site_html):
    for site, h in site_html.items():
        prof = h[h.find("Respondent profile"):h.find("Data source")]
        cells = re.findall(r"<td class='num'>(\d+)</td><td class='num'>\d+%</td>", prof)
        assert all(int(c) >= R.SMALL_COUNT for c in cells), site
    assert any("&lt;5" in h for h in site_html.values())


def test_all_comments_screen_only(site_html):
    for h in site_html.values():
        assert "<section class='screen-only'><h2>All comments by category" in h
        assert "print-only" in h


def test_pptx_matches_card_numbers(ctx):
    import io
    from pptx import Presentation
    site = "GA-Atlanta"
    d = R.site_report_data(ctx, site)
    prs = Presentation(io.BytesIO(R.build_site_pptx(ctx, [site])))
    txt = " ".join(sh.text_frame.text for sh in prs.slides[0].shapes if sh.has_text_frame)
    cells = " ".join(c.text for sh in prs.slides[0].shapes if sh.has_table for row in sh.table.rows for c in row.cells)
    assert R.fmt_margin(d["card"]["overall"], d["card"]["margin"]) in txt
    assert R.standing_text(d["card"]["standing"]) in txt
    for t in d["top_items"]:
        assert t["status_words"] in cells and f"{t['friction']:.0f}" in cells
