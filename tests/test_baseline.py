"""Section 4 of the Nov 2026 review brief: numbers that must not move.

Completed responses, 10 in-scope sites. These reconcile with the official
Tableau deck; any change to them is a regression.
"""
import pandas as pd
import pytest

import reports as R

INDEX_BY_SITE = {
    "GA-Atlanta": 76.54, "FL-Miami": 71.70, "NY-New York (Penn1)": 71.35,
    "IL-Chicago-Willis Tower": 70.24, "VA-Norfolk": 69.78, "OH-Mason": 68.44,
    "IN-Indianapolis": 68.17, "FL-Tampa": 67.76, "VA-Richmond": 66.54,
    "CA-Woodland Hills": 65.75,
}
DIMENSIONS = {"Work Activity": 62.5, "Physical Feature": 66.1,
              "Service Level": 75.6, "Digital & Tech": 72.0}


@pytest.fixture(scope="module")
def ctx():
    return R.load_context(completers_only=True)


def test_index_by_site(ctx):
    overall = R.site_standings(ctx)["overall"]
    assert set(overall) == set(INDEX_BY_SITE)
    for site, want in INDEX_BY_SITE.items():
        assert overall[site] == pytest.approx(want, abs=0.01), site


def test_company_index_and_dimensions(ctx):
    sd = ctx.scores_dim.set_index("dimension")["composite"].astype(float)
    assert sd["OVERALL (equal)"] == pytest.approx(69.0, abs=0.05)
    for dim, want in DIMENSIONS.items():
        assert sd[dim] == pytest.approx(want, abs=0.05), dim


def test_status_counts(ctx):
    ss = ctx.scores_site
    imp = ss[pd.to_numeric(ss["mean_importance"], errors="coerce").notna()]
    assert len(imp) == 235
    counts = imp["ryg"].value_counts().to_dict()
    assert counts == {"Under Supported": 76, "Target": 148, "Over Supported": 11}
