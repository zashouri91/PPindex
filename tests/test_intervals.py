"""Phase 3.1: bootstrap intervals and paired-gap columns written by build_data.py."""
import io
import os
import subprocess

import pandas as pd
import pytest

import reports as R
from conftest import ROOT

MEASURES = ["Index", "Work Activity", "Physical Feature", "Service Level", "Digital & Tech"]


@pytest.mark.parametrize("fn,flag", [("site_intervals.csv", True), ("site_intervals_all.csv", False)])
def test_interval_files(fn, flag):
    iv = pd.read_csv(os.path.join(ROOT, "data", fn))
    ctx = R.load_context(flag)
    stand = R.site_standings(ctx)
    for site in ctx.meta["ten_sites"]:
        sub = iv[iv["site"] == site].set_index("measure")
        assert list(sub.index) == MEASURES
        assert (sub["lo"] <= sub["estimate"]).all() and (sub["estimate"] <= sub["hi"]).all()
        assert (sub["n_boot"] == 2000).all()
        # same method as site_standings (which reads 1-dp item means)
        assert sub.loc["Index", "estimate"] == pytest.approx(stand["overall"][site], abs=0.06)
    assert "Company (10 sites)" in set(iv["site"])


def test_gap_columns_appended_without_touching_existing():
    for fn in ("scores_by_site.csv", "scores_overall.csv"):
        new = pd.read_csv(os.path.join(ROOT, "data", fn))
        assert list(new.columns[-5:]) == ["gap", "gap_se", "gap_lo", "gap_hi", "support_se"]
        imp = new[new["mean_importance"].notna()]
        g = imp["mean_support"] - imp["mean_importance"]
        assert (abs(imp["gap"] - g) < 0.002).all()        # paired gap = difference of means
        assert ((imp["gap_lo"] <= imp["gap"]) & (imp["gap"] <= imp["gap_hi"])).all()
        assert new[new["mean_importance"].isna()]["gap"].isna().all()
