"""Gold set file and evaluation harness."""
import os

import pandas as pd

from conftest import ROOT
from taxonomy import THEMES
import tools.eval_tagging as EV

GOLD = os.path.join(ROOT, "data", "gold", "gold_labels.csv")


def test_gold_file_holds_ids_and_labels_only():
    g = pd.read_csv(GOLD)
    assert list(g.columns) == ["comment_id", "themes", "sentiment", "labeler", "verified", "notes"]
    assert len(g) == 300 and g["comment_id"].is_unique
    c = pd.read_csv(os.path.join(ROOT, "data", "comments.csv"))
    assert set(g["comment_id"]) <= set(c["comment_id"])
    valid = set(THEMES) | {"Unclassified"}
    for s in g["themes"].dropna():
        assert set(s.split("|")) <= valid, s
    assert set(g["sentiment"].dropna()) <= set(EV.SENTS)
    # notes must never carry comment text
    texts = set(c.set_index("comment_id").loc[g["comment_id"], "comment"].astype(str))
    assert not any(n in texts for n in g["notes"].dropna().astype(str) if len(n) > 20)


def test_score_toy():
    gold = pd.DataFrame({"comment_id": [1, 2, 3], "themes": ["Lighting", "Lighting|Noise & Distraction", "Unclassified"],
                         "sentiment": ["Negative", "Negative", "Positive"], "verified": [True] * 3})
    pred = pd.DataFrame({"comment_id": [1, 2, 3], "themes": ["Lighting", "Lighting", "Lighting"],
                         "sentiment": ["Neutral / Mixed", "Negative", "Positive"]})
    r = EV.score(gold, pred)
    T = r["themes"].set_index("theme")
    assert T.loc["Lighting", "precision"] == 2 / 3 and T.loc["Lighting", "recall"] == 1.0
    assert T.loc["Noise & Distraction", "recall"] == 0.0
    assert r["sent_acc"] == 2 / 3 and r["more_positive"] == 1 and r["more_negative"] == 0
