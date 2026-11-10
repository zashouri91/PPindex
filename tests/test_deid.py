"""Guard: no tracked file under data/ may carry an identifying column."""
import io
import os
import subprocess

import pandas as pd
import pytest

from conftest import ROOT
from deid import PII_EXACT, drop_pii_columns, is_pii_column


def _tracked_data_files():
    try:
        out = subprocess.check_output(["git", "ls-files", "data"], cwd=ROOT, text=True)
        return [f for f in out.split() if f]
    except Exception:  # not a git checkout: fall back to what's on disk
        files = []
        for dp, _dn, fns in os.walk(os.path.join(ROOT, "data")):
            files += [os.path.relpath(os.path.join(dp, f), ROOT) for f in fns]
        return [f for f in files if not f.endswith("raw.xlsx")]


def _columns(path):
    full = os.path.join(ROOT, path)
    if path.endswith(".csv"):
        return list(pd.read_csv(full, nrows=0).columns)
    if path.endswith((".xlsx", ".xls")):
        cols = []
        for _name, df in pd.read_excel(full, sheet_name=None, nrows=0).items():
            cols += list(df.columns)
        return cols
    return []


def test_original_raw_is_not_tracked():
    assert "data/raw.xlsx" not in _tracked_data_files()


@pytest.mark.parametrize("path", [f for f in _tracked_data_files()
                                  if f.endswith((".csv", ".xlsx", ".xls"))])
def test_no_identifying_columns(path):
    bad = [c for c in _columns(path) if is_pii_column(c)]
    assert not bad, f"{path} carries identifying column(s): {bad}"


def test_matching_is_case_insensitive():
    for name in PII_EXACT:
        assert is_pii_column(name.upper()) and is_pii_column(f"  {name.lower()} ")
    assert is_pii_column("5. If willing to be contacted, enter your us DOMAIN id")
    assert not is_pii_column("Respondent")
    assert not is_pii_column("Reached end")
    assert not is_pii_column("Completion time (seconds)")


def test_drop_pii_columns_keeps_order():
    df = pd.DataFrame(columns=["Respondent", "IP", "Q1", "browser", "Q2"])
    out = drop_pii_columns(df, notice=None)
    assert list(out.columns) == ["Respondent", "Q1", "Q2"]
