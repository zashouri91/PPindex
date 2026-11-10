"""
Optional model-based comment tagger — OFF by default.

The keyword rules in taxonomy.py remain the default. build_data.py uses this
module only when PULSEPOINT_TAGGER=llm. It should be switched on only if it
measurably beats the rules on VERIFIED gold labels (tools/eval_tagging.py).

Per comment the model returns JSON at temperature 0:
    {"themes": [...one or more of the 29 taxonomy themes...],
     "sentiment": "Negative" | "Neutral / Mixed" | "Positive",
     "intent": "complaint" | "request" | "praise" | "neutral"}
Anything outside those sets is dropped or mapped; an empty theme list becomes
"Unclassified". Results are cached by (comment_id, taxonomy version, model) in
data/llm_cache/ (labels only, gitignored), so a rebuild re-asks only for new
comments or after the taxonomy or model changes.

PRIVACY: comment text goes only to the local Ollama backend unless
PULSEPOINT_ALLOW_HOSTED_COMMENTS=1 (enforced in llm_backends.call).

CLI (for evaluation):
    python llm_tagger.py --gold --export data/llm_cache/llm_gold_predictions.csv
    python tools/eval_tagging.py --pred llm=data/llm_cache/llm_gold_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import pandas as pd

import llm_backends as LB
from taxonomy import THEMES, TAXONOMY_VERSION, is_junk

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "data", "llm_cache")
SENTIMENTS = ["Negative", "Neutral / Mixed", "Positive"]
INTENTS = ["complaint", "request", "praise", "neutral"]
THEME_NAMES = list(THEMES)

SYSTEM_PROMPT = (
    "You label workplace-survey comments. Reply with ONE JSON object only, no prose:\n"
    '{"themes": [..], "sentiment": "..", "intent": ".."}\n'
    "themes: every theme the comment is ABOUT, chosen ONLY from this exact list (copy the names exactly); "
    "use [] if none fits. A passing mention is not enough — e.g. a desk mentioned while complaining about "
    "noise is not 'Desk Availability & Reservations' (that theme needs booking, availability or assignment of "
    "desks/spaces); 'Microsoft Teams' the app is not 'AV / Video Conferencing' unless audio, video or "
    "room technology is the issue; 'Collaboration Space' needs a physical space; colleagues being elsewhere "
    "is 'In-Person / Team Presence'.\n"
    "THEMES:\n- " + "\n- ".join(THEME_NAMES) + "\n"
    "sentiment: Negative, Neutral / Mixed, or Positive. A complaint OR a request for something missing "
    "('would be nice to have more...', 'we need...') is Negative. Neutral / Mixed only when there is no "
    "evaluation or genuine praise and complaint carry equal weight.\n"
    "intent: complaint, request, praise, or neutral."
)


def _cache_path(model: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", f"{model}_{TAXONOMY_VERSION}").strip("_").lower()
    return os.path.join(CACHE_DIR, f"tagger_{slug}.json")


def load_cache(model: str) -> dict:
    try:
        with open(_cache_path(model)) as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_cache(model: str, cache: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_path(model), "w") as fh:
        json.dump(cache, fh)


def _key(comment_id, model) -> str:
    return f"{int(comment_id)}|{TAXONOMY_VERSION}|{model}"


def normalise(obj) -> dict:
    """Coerce a model reply into the allowed label sets."""
    if not isinstance(obj, dict):
        obj = {}
    lower = {t.lower(): t for t in THEME_NAMES}
    themes = []
    for t in obj.get("themes") or []:
        name = lower.get(str(t).strip().lower())
        if name and name not in themes:
            themes.append(name)
    s = str(obj.get("sentiment", "")).strip().lower()
    sentiment = ("Negative" if s.startswith("neg") else "Positive" if s.startswith("pos")
                 else "Neutral / Mixed")
    intent = str(obj.get("intent", "")).strip().lower()
    intent = intent if intent in INTENTS else "neutral"
    return {"themes": themes or ["Unclassified"], "sentiment": sentiment, "intent": intent}


def _parse(text: str) -> dict:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        a, b = t.find("{"), t.rfind("}")
        return json.loads(t[a:b + 1]) if a != -1 and b > a else {}


def tag_one(text: str, question_label: str, provider: str, model: str, call=None) -> dict:
    call = call or LB.call
    user = f"Survey question: {question_label}\nComment: {text}"
    raw = call(provider, SYSTEM_PROMPT, user, model, sends_comment_text=True)
    return normalise(_parse(raw))


def tag_comments(df: pd.DataFrame, provider: str = None, model: str = None, call=None,
                 progress=None) -> pd.DataFrame:
    """Tag the non-junk rows of df (columns comment_id, comment, source_question).
    Returns comment_id, themes ('|'-joined), sentiment, intent. Uses / fills the cache."""
    provider = provider or LB.configured_provider() or "ollama"
    model = model or LB.model_name(provider)
    if call is None:
        LB.require_local_or_opt_in(provider, sends_comment_text=True)
    cache = load_cache(model)
    out, dirty = [], 0
    rows = df[~df["comment"].map(is_junk)]
    for i, (_, r) in enumerate(rows.iterrows()):
        k = _key(r["comment_id"], model)
        if k not in cache:
            cache[k] = tag_one(str(r["comment"]), str(r.get("source_question", "")), provider, model, call)
            dirty += 1
            if dirty % 25 == 0:
                save_cache(model, cache)
        res = cache[k]
        out.append({"comment_id": int(r["comment_id"]), "themes": "|".join(res["themes"]),
                    "sentiment": res["sentiment"], "intent": res["intent"]})
        if progress:
            progress(i + 1, len(rows))
    if dirty:
        save_cache(model, cache)
    return pd.DataFrame(out, columns=["comment_id", "themes", "sentiment", "intent"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", action="store_true", help="tag only the gold-set comments")
    ap.add_argument("--export", required=True, help="CSV path for comment_id, themes, sentiment, intent")
    a = ap.parse_args(argv)
    c = pd.read_csv(os.path.join(HERE, "data", "comments.csv"))
    if a.gold:
        ids = set(pd.read_csv(os.path.join(HERE, "data", "gold", "gold_labels.csv"))["comment_id"])
        c = c[c["comment_id"].isin(ids)]
    res = tag_comments(c, progress=lambda i, n: print(f"\r{i}/{n}", end="", file=sys.stderr))
    os.makedirs(os.path.dirname(os.path.abspath(a.export)), exist_ok=True)
    res.to_csv(a.export, index=False)
    print(f"\nwrote {len(res)} rows to {a.export}")


if __name__ == "__main__":
    main()
