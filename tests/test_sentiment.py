"""Sentiment: the question-framing prior is keyed by question label. Invented text only."""
import json
import os

import pytest

from conftest import ROOT
from sentiment import QUESTION_PRIOR, score_sentiment

TEXT = "The third floor has a printer and two kitchens."   # invented, near-neutral


def test_prior_applies_by_question_label():
    base, _ = score_sentiment(TEXT)
    for label, prior in QUESTION_PRIOR.items():
        s, _ = score_sentiment(TEXT, label)
        assert s == pytest.approx(base + prior, abs=1e-3), label
    assert score_sentiment(TEXT, "Physical Limitations")[0] < base


def test_domain_names_are_not_prior_keys():
    # the bug: build_data passed the question DOMAIN, which never matched
    for domain in ("Physical Environment", "Focused / Private Work"):
        assert score_sentiment(TEXT, domain)[0] == pytest.approx(score_sentiment(TEXT)[0])


def test_prior_keys_match_built_question_labels():
    with open(os.path.join(ROOT, "data", "meta.json")) as fh:
        questions = set(json.load(fh)["questions"])
    assert set(QUESTION_PRIOR) == questions


def test_build_data_passes_the_label():
    src = open(os.path.join(ROOT, "build_data.py")).read()
    assert "sentiment(text, qlabel)" in src and "sentiment(text, qdomain)" not in src
