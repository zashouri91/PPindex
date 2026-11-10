"""
Score the comment tagging against the gold labels.

    python tools/eval_tagging.py [--pred NAME=path.csv ...] [--out data/gold/eval_report.md]

Default prediction source is "rules": the keyword taxonomy + sentiment model
as stored in data/comments.csv (themes, sentiment_label). Extra sources are
CSVs with columns comment_id, themes, sentiment (e.g. the model tagger's
output from `python llm_tagger.py --export`).

Reports, SEPARATELY for verified labels (verified == True) and for draft labels
(not yet verified): per-theme precision / recall / F1, micro and macro F1,
sentiment accuracy, the confusion matrix, and the direction of sentiment
errors (prediction more positive / more negative than the label). Writes
metrics only — never comment text.
"""
import argparse
import os
import sys
from datetime import date

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from taxonomy import THEMES, TAXONOMY_VERSION  # noqa: E402

GOLD = os.path.join(ROOT, "data", "gold", "gold_labels.csv")
OUT = os.path.join(ROOT, "data", "gold", "eval_report.md")
SENTS = ["Negative", "Neutral / Mixed", "Positive"]
ORDER = {s: i for i, s in enumerate(SENTS)}
THEME_LIST = list(THEMES) + ["Unclassified"]


def _set(s):
    return {t.strip() for t in str(s).split("|") if t.strip() and str(s) != "nan"} or {"Unclassified"}


def _truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def load_gold(path=GOLD):
    g = pd.read_csv(path)
    g = g[g["themes"].notna() & (g["themes"].astype(str).str.strip() != "")]
    g["verified"] = g["verified"].map(_truthy)
    return g


def load_rules():
    c = pd.read_csv(os.path.join(ROOT, "data", "comments.csv"))
    return c[["comment_id", "themes", "sentiment_label"]].rename(columns={"sentiment_label": "sentiment"})


def score(gold: pd.DataFrame, pred: pd.DataFrame) -> dict:
    m = gold.merge(pred, on="comment_id", how="inner", suffixes=("_gold", "_pred"))
    out = {"n": len(m)}
    if not len(m):
        return out
    G = m["themes_gold"].map(_set)
    P = m["themes_pred"].map(_set)
    rows, tp_all, fp_all, fn_all = [], 0, 0, 0
    for th in THEME_LIST:
        tp = sum(th in g and th in p for g, p in zip(G, P))
        fp = sum(th not in g and th in p for g, p in zip(G, P))
        fn = sum(th in g and th not in p for g, p in zip(G, P))
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if tp and (prec + rec) else 0.0
        rows.append({"theme": th, "gold": tp + fn, "pred": tp + fp, "tp": tp,
                     "precision": prec, "recall": rec, "f1": f1})
    T = pd.DataFrame(rows)
    out["themes"] = T
    mp = tp_all / (tp_all + fp_all) if tp_all + fp_all else float("nan")
    mr = tp_all / (tp_all + fn_all) if tp_all + fn_all else float("nan")
    out["micro"] = {"precision": mp, "recall": mr, "f1": 2 * mp * mr / (mp + mr) if mp + mr else 0.0}
    out["macro_f1"] = float(T[T["gold"] > 0]["f1"].mean()) if len(T) else float("nan")
    s = m[m["sentiment_gold"].isin(SENTS) & m["sentiment_pred"].isin(SENTS)]
    out["sent_n"] = len(s)
    out["sent_acc"] = float((s["sentiment_gold"] == s["sentiment_pred"]).mean()) if len(s) else float("nan")
    out["confusion"] = pd.crosstab(pd.Categorical(s["sentiment_gold"], SENTS),
                                   pd.Categorical(s["sentiment_pred"], SENTS),
                                   rownames=["label"], colnames=["predicted"], dropna=False)
    d = s["sentiment_pred"].map(ORDER) - s["sentiment_gold"].map(ORDER)
    out["more_positive"], out["more_negative"] = int((d > 0).sum()), int((d < 0).sum())
    return out


def _fmt(v):
    return "—" if pd.isna(v) else f"{v:.0%}"


def section(title, res):
    if res.get("n", 0) == 0:
        return f"### {title}\n\nNo labels in this group yet.\n"
    lines = [f"### {title}", "",
             f"Comments scored: {res['n']}. Themes — micro precision {_fmt(res['micro']['precision'])}, "
             f"recall {_fmt(res['micro']['recall'])}, F1 {_fmt(res['micro']['f1'])}; macro F1 {_fmt(res['macro_f1'])}.",
             f"Sentiment accuracy {_fmt(res['sent_acc'])} on {res['sent_n']} comments; of the errors, "
             f"{res['more_positive']} were more positive than the label and {res['more_negative']} more negative.", "",
             "| Theme | Labelled | Predicted | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|"]
    for _, r in res["themes"].sort_values("gold", ascending=False).iterrows():
        lines.append(f"| {r['theme']} | {r['gold']} | {r['pred']} | {_fmt(r['precision'])} | "
                     f"{_fmt(r['recall'])} | {_fmt(r['f1'])} |")
    cm = res["confusion"]
    lines += ["", "Sentiment confusion (rows = label, columns = predicted):", "",
              "| label \\ predicted | " + " | ".join(cm.columns) + " |",
              "|---|" + "---:|" * len(cm.columns)]
    for lab, r in cm.iterrows():
        lines.append(f"| {lab} | " + " | ".join(str(int(v)) for v in r.values) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", default=[], help="NAME=path.csv (comment_id, themes, sentiment)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    gold = load_gold()
    sources = {"rules": load_rules()}
    for spec in a.pred:
        name, path = spec.split("=", 1)
        sources[name] = pd.read_csv(path)
    ver, draft = gold[gold["verified"]], gold[~gold["verified"]]
    md = [f"# Comment tagging — accuracy against the gold set", "",
          f"Generated {date.today().isoformat()} by tools/eval_tagging.py · taxonomy {TAXONOMY_VERSION}. "
          f"Gold set: {len(gold)} labelled comments ({len(ver)} verified by a person, {len(draft)} draft). "
          "Convention: a complaint or an unmet-need request is Negative. Draft labels were pre-filled by an "
          "AI assistant and are NOT a validation — only verified rows count as evidence of accuracy.", ""]
    for name, pred in sources.items():
        md += [f"## Source: {name}", "", section("Verified labels", score(ver, pred)),
               section("Draft labels (unverified)", score(draft, pred))]
    with open(a.out, "w") as fh:
        fh.write("\n".join(md))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
