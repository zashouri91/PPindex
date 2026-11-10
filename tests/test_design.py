"""Phase 7: colour contrast, one meaning per colour, minimum text sizes."""
import re

import pytest

import action_plan
import design
import reports as R


def test_css_variables_meet_aa():
    css = design.css_vars()
    v = design.parse_css_vars(css)
    failures = []
    for fg, bg, minimum in design.CONTRAST_PAIRS:
        ratio = design.contrast(v[fg], v[bg])
        if ratio < minimum:
            failures.append(f"{fg} on {bg}: {ratio:.2f} < {minimum}")
    assert not failures, failures


def test_contrast_math():
    assert design.contrast("#000000", "#ffffff") == pytest.approx(21.0, abs=0.01)
    assert design.contrast("#777777", "#ffffff") == pytest.approx(4.48, abs=0.02)


def _screen_css(css):
    # drop @media print{...} blocks (one level of nesting)
    return re.sub(r"@media print\s*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}", "", css)


@pytest.fixture(scope="module")
def rendered():
    ctx = R.load_context(True)
    return [R.build_site_report(ctx, "VA-Richmond"), R.build_portfolio_report(ctx)]


def _styles(html):
    return "".join(re.findall(r"<style>(.*?)</style>", html, re.S))


def test_card_css_uses_variables_only(rendered):
    for html in rendered:
        css = _styles(html)
        body = css.replace(design.css_vars(), "")
        hexes = re.findall(r"#[0-9a-fA-F]{3,6}\b", body)
        assert not hexes, f"hard-coded colours outside the variables: {sorted(set(hexes))}"


def test_action_plan_css_has_no_status_colours():
    assert not re.findall(r"#[0-9a-fA-F]{3,6}\b", action_plan.ACTION_PLAN_CSS)
    assert "--st-" not in action_plan.ACTION_PLAN_CSS     # status colours are for support status only


def test_minimum_font_sizes(rendered):
    for html in rendered:
        css = _styles(html)
        screen = _screen_css(css)
        sizes = [float(s) for s in re.findall(r"font-size:\s*([\d.]+)px", screen)]
        assert min(sizes) >= design.MIN_NOTE_PX_SCREEN, sorted(sizes)[:3]
        printed = [float(s) for blk in re.findall(r"@media print\s*\{(.*?)\}\s*\}", css, re.S)
                   for s in re.findall(r"font-size:\s*([\d.]+)px", blk)]
        assert not printed or min(printed) >= design.MIN_PRINT_TABLE_PX


def test_no_inline_hex_colours(rendered):
    for html in rendered:
        body = re.sub(r"<style>.*?</style>", "", html, flags=re.S)
        assert not re.findall(r"style='[^']*#[0-9a-fA-F]{3,6}", body)
