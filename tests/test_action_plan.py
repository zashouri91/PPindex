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


def _toy_payload():
    return {"overall_index_score_0_100_unweighted": 68.2,
            "highest_friction_items": [{"item": "Thermal", "friction_0_100": 23, "pct_under_supported": 46}],
            "candidate_priority_themes": [{"theme": "Temperature / Thermal", "people": 117, "comments": 130,
                                           "pct_negative": 77}],
            "representative_comments": [{"theme": "Temperature / Thermal",
                                         "text": "The fourth floor is freezing every afternoon and the vents blow on my desk."}]}


def _toy_plan(evidence, looks):
    return {"priorities": [{"name": "Thermal comfort", "evidence": [{"type": "Finding", "text": evidence}],
                            "whatItLooksLike": looks}]}


def test_grounding_passes_when_numbers_and_quotes_trace():
    plan = _toy_plan("Temperature / Thermal: 117 people, 130 comments, 77% negative; friction 23/100 (46% under-supported).",
                     ["The fourth floor is freezing every afternoon ... the vents blow on my desk."])
    v = action_plan.validate_grounding(plan, _toy_payload())
    assert v["ok"], v["failures"]


def test_grounding_rejects_invented_numbers_and_quotes():
    plan = _toy_plan("Temperature / Thermal: 375 comments, 80% negative.",
                     ["People say the building is an icebox all year."])
    v = action_plan.validate_grounding(plan, _toy_payload())
    assert not v["ok"]
    assert any("375" in f for f in v["failures"]) and any("what it looks like" in f for f in v["failures"])


def test_unvalidated_plan_is_not_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(action_plan, "PLANS_DIR", str(tmp_path))
    bad = _toy_plan("x", [])
    bad["_meta"] = {"validation": {"ok": False, "failures": ["P1 evidence: number 9"]}}
    with pytest.raises(ValueError):
        action_plan.save_plan("Test Site", bad)
    bad["_meta"]["validation"]["ok"] = True
    action_plan.save_plan("Test Site", bad)


def test_payload_has_standing_severity_and_survey_items(ctx):
    d = R.site_report_data(ctx, "IN-Indianapolis")
    p = action_plan.build_report_card_payload(d, include_verbatims=True)
    assert p["overall_index_standing"] == "In line with the company average"
    assert p["top_survey_items"] and p["top_survey_items"][0]["severity"] in {"slight", "moderate", "severe"}
    assert all("severity" in h for h in p["highest_friction_items"])
    assert "tenure" not in str(p["representative_comments"])
    roles_ok = R.attributable_roles(ctx, "IN-Indianapolis")
    assert all(q["role"] in roles_ok | {""} for q in p["representative_comments"])


def test_prompt_v32_rules():
    pr = action_plan.build_prompt("Owner")
    assert "Never rank sites" in pr and "PRIORITIES START FROM THESE ITEMS" in pr
    assert "v3.2" in action_plan.PROMPT_VERSION and "sign-off" in action_plan.PROMPT_SIGNOFF_NOTE


def test_model_defaults_current():
    import llm_backends as LB
    assert LB.model_name("anthropic") == "claude-sonnet-5-5"
    assert LB.model_name("openai") == "gpt-6.1-sol"
    assert "latest" not in LB.model_name("anthropic")


def test_stored_plans_refreshed():
    import glob, json, re
    for f in glob.glob(action_plan.PLANS_DIR + "/*.json"):
        p = json.load(open(f))
        assert "indexTier" not in p and p["_meta"].get("evidence_refreshed_at")
        own = json.dumps([p.get("assumptions"), p.get("protectWhatIsWorking"),
                          [e for pr in p["priorities"] for e in pr["evidence"]]])
        assert not re.search(r"rank \d|\d+ of 10\b|\bthird\b|of all sites", own), f
