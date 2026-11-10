"""Model backends and the optional tagger. No network: every model call is stubbed."""
import json

import pandas as pd
import pytest

import action_plan
import llm_backends as LB
import llm_tagger


def test_hosted_comment_text_needs_opt_in(monkeypatch):
    monkeypatch.delenv(LB.HOSTED_OPT_IN, raising=False)
    LB.require_local_or_opt_in("ollama", sends_comment_text=True)          # local: fine
    LB.require_local_or_opt_in("openai", sends_comment_text=False)         # aggregate-only: fine
    for p in ("openai", "azure", "anthropic"):
        with pytest.raises(LB.HostedCommentsNotAllowed):
            LB.require_local_or_opt_in(p, sends_comment_text=True)
    monkeypatch.setenv(LB.HOSTED_OPT_IN, "1")
    LB.require_local_or_opt_in("anthropic", sends_comment_text=True)


def test_call_checks_privacy_before_any_request(monkeypatch):
    monkeypatch.delenv(LB.HOSTED_OPT_IN, raising=False)
    hit = []
    monkeypatch.setitem(LB.BACKENDS, "openai", lambda *a: hit.append(a) or "{}")
    with pytest.raises(LB.HostedCommentsNotAllowed):
        LB.call("openai", "sys", "user", "m", sends_comment_text=True)
    assert hit == []


def test_action_plan_refuses_hosted_verbatims(monkeypatch):
    monkeypatch.delenv(LB.HOSTED_OPT_IN, raising=False)
    monkeypatch.setenv("PULSEPOINT_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-key")
    monkeypatch.setitem(LB.BACKENDS, "anthropic", lambda *a: pytest.fail("network call attempted"))
    with pytest.raises(LB.HostedCommentsNotAllowed):
        action_plan.generate_action_plan({"site": "X"}, include_verbatims=True)


def test_ollama_is_the_default(monkeypatch):
    monkeypatch.delenv("PULSEPOINT_LLM_PROVIDER", raising=False)
    assert LB.configured_provider() == "ollama"


def test_normalise_keeps_only_taxonomy_themes():
    r = llm_tagger.normalise({"themes": ["lighting", "Made Up Theme", "Lighting"],
                              "sentiment": "negative", "intent": "Request"})
    assert r == {"themes": ["Lighting"], "sentiment": "Negative", "intent": "request"}
    assert llm_tagger.normalise({})["themes"] == ["Unclassified"]
    assert llm_tagger.normalise({"sentiment": "mixed"})["sentiment"] == "Neutral / Mixed"


def test_tag_comments_uses_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_tagger, "CACHE_DIR", str(tmp_path))
    calls = []

    def stub(provider, system, user, model, sends_comment_text):
        calls.append(user)
        assert sends_comment_text and "Lighting" in system
        return json.dumps({"themes": ["Lighting"], "sentiment": "Negative", "intent": "complaint"})

    df = pd.DataFrame({"comment_id": [1, 2, 3], "source_question": ["Q"] * 3,
                       "comment": ["The lights flicker all day long.", "no", "Too dim near the windows."]})
    out = llm_tagger.tag_comments(df, provider="ollama", model="stub-model", call=stub)
    assert list(out["comment_id"]) == [1, 3] and len(calls) == 2          # junk skipped
    llm_tagger.tag_comments(df, provider="ollama", model="stub-model", call=stub)
    assert len(calls) == 2                                                # served from cache
    llm_tagger.tag_comments(df, provider="ollama", model="other-model", call=stub)
    assert len(calls) == 4                                                # cache is per model


def test_build_data_uses_rules_unless_opted_in():
    src = open(llm_tagger.HERE + "/build_data.py").read()
    assert 'os.environ.get("PULSEPOINT_TAGGER", "rules")' in src
