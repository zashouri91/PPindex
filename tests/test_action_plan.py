"""Action-plan payload and helpers (no model calls)."""
import pytest

import action_plan
import reports as R


@pytest.fixture(scope="module")
def ctx():
    return R.load_context(completers_only=True)


def test_payload_respondents_are_site_counts(ctx):
    site = "VA-Norfolk"
    d = R.site_report_data(ctx, site)
    p = action_plan.build_report_card_payload(d, include_verbatims=False)
    n_site = int((ctx.resp_scores["site"] == site).sum())
    n_all = int(ctx.resp_scores["site"].isin(ctx.meta["ten_sites"]).sum())
    assert p["respondents_at_site"] == n_site
    assert p["respondents_all_10_sites"] == n_all
    assert n_site < n_all
