"""Round 2 (second wave): built separately, benchmarked against its own sites, and
never allowed to move a round 1 number. No comment text appears in these tests."""
import os

import numpy as np
import pandas as pd
import pytest

import reports as R
import waves
from deid import pii_columns

R2_DIR = waves.WAVES["round2"]["data_dir"]
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(R2_DIR, "meta.json")),
                                reason="round 2 not built")


@pytest.fixture(scope="module")
def c1():
    return R.load_context(True, "round1")


@pytest.fixture(scope="module")
def c2():
    return R.load_context(True, "round2")


def test_round2_raw_is_deidentified():
    cols = pd.read_excel(waves.WAVES["round2"]["raw"], nrows=0).columns
    assert pii_columns(cols) == []


def test_site_name_normalisation():
    f = waves.normalise_round2_site
    assert f("CT-WALLINGFORD, 1 EXAMPLE RD") == "CT-Wallingford"
    assert f("NV-LAS VEGAS, 9133 W RUSSELL RD") == "NV-Las Vegas-Russell"
    assert f("NV-LAS VEGAS, 3634 S MARYLAND PKWY") == "NV-Las Vegas-Maryland Pkwy"
    assert f("IL-CHICAGO, 8600 W BRYN MAWR AVE") == "IL-Chicago-Bryn Mawr"
    assert f("Already-Short") == "Already-Short"


def test_round2_scope_and_wave_meta(c2):
    sites = c2.meta["ten_sites"]
    assert len(sites) == 30 and all("," not in s for s in sites)
    assert not set(sites) & set(waves.ROUND1_SITES)
    w = c2.meta["wave"]
    assert w["key"] == "round2" and w["company_row"] == "Round 2 (30 sites)"
    assert set(c2.intervals["site"]) == set(sites) | {"Round 2 (30 sites)"}


def test_round1_meta_untouched_in_substance(c1):
    assert c1.meta["ten_sites"] == waves.ROUND1_SITES
    assert c1.meta["wave"]["key"] == "round1"
    assert c1.meta["items_not_asked"] == []
    assert R.company_value(c1, "Index") == 69.0


def test_unasked_items_are_left_out_not_given_a_status(c2):
    na = {i["item"] for i in c2.meta["items_not_asked"]}
    assert na == {"Ability to get help resolving workplace or technology issues while on site",
                  "Support from a community manager"}
    assert not set(c2.scores["item"]) & na
    assert c2.scores["n"].min() > 0
    assert (c2.scores_site["n"] > 0).all()


def test_round2_site_card_uses_round2_benchmark(c2):
    small = min(c2.meta["ten_sites"], key=lambda s: (c2.resp_scores["site"] == s).sum())
    big = max(c2.meta["ten_sites"], key=lambda s: (c2.resp_scores["site"] == s).sum())
    for site in (small, big):
        h = R.build_site_report(c2, site)
        body = h.split("</style>", 1)[1]
        assert "round 2 avg" in body and "the round 2 average" in body
        assert "the company average" not in body
        assert "id='plan'" not in body                       # no stored plans for round 2
        assert "Not asked in Round 2" in body
    assert "Small sample:" in R.build_site_report(c2, small)
    assert "Small sample:" not in R.build_site_report(c2, big)


def test_round2_portfolio(c2):
    h = R.build_portfolio_report(c2)
    assert "Round 2 PulsePoints — Portfolio Report Card" in h
    assert h.count("<table class='qgrid'>") == 2            # 30 sites split into two grids
    assert "For the 22 items with an Importance rating" in h
    assert "small sample, n=" in h


def test_round1_cards_unchanged(c1):
    h = R.build_site_report(c1, "GA-Atlanta")
    assert "the company average" in h and "round 2" not in h.lower().split("</style>", 1)[1]
    p = R.build_portfolio_report(c1)
    assert "10 PulsePoints — Portfolio Report Card" in p and "For the 24 items" in p


def test_combined_like_for_like(c1, c2):
    import combined_report as C
    allc = [c for c in C._item_support(c1).columns if "|" in c]
    full = C.matched_round_scores(c1, allc)                 # all items -> official round 1 values
    for m in ["Index"] + C.DIMS:
        assert abs(full[m]["est"] - R.company_value(c1, m)) < 0.06
    d = C.combined_data(c1, c2)
    assert len(d["dropped_items"]) == 2
    assert {s["round"] for s in d["sites"]} == {"Round 1", "Round 2"} and len(d["sites"]) == 40
    for r in d["scores"] + d["sentiment"]:
        assert np.isfinite(r["margin"])
    h = C.build_combined_portfolio(c1, c2)
    assert "different sites" in h and "Like for like" in h
