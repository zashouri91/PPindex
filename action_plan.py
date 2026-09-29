"""
Operational Action Plan — "What to Do Next" generator for PulsePoint site cards.

Turns a single site's report-card evidence into a validated top-5 set of
operational recommendations, following the v3 master prompt agreed with the
work-stream team. The design constraints from that review are baked in here:

  * GROUNDED    — the model sees ONLY this site's real evidence (Index score,
                  dimension scores, the computed priority ranking, per-theme
                  stats, highest-Friction items, and the actual representative
                  quotes). No other location; nothing invented.
  * REPRODUCIBLE— temperature 0; generate ONCE and store the JSON per site; the
                  card renders from the stored copy, never live on every view.
  * PRIVACY-1ST — the LLM backend is pluggable and the DEFAULT is a LOCAL Ollama
                  model, so verbatim employee comments never leave the machine.
                  include_verbatims=False sends aggregate-only if policy requires.
  * OWNER-CONFIG— the primary-owner label is a single variable (env-overridable
                  and editable per generation), because that org decision is open.
  * UNWEIGHTED  — only the equal/unweighted Index Score (the Tableau-matching
                  headline) is ever sent; the weighted variant is never exposed.

Importing this module never touches the network. generate_action_plan() is the
only function that calls a model, and only when explicitly invoked.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

try:
    from taxonomy import THEME_TO_CATEGORY
except Exception:  # taxonomy always ships alongside; fall back to empty map
    THEME_TO_CATEGORY = {}

# ---------------------------------------------------------------------------
# Config (all read at call time from the environment, never at import)
# ---------------------------------------------------------------------------
DEFAULT_OWNER = "Regional Director + Local Team"

# Themes that are generic keyword catch-alls with no specific, actionable
# content. Excluded from every input the model sees (recommendation #4 /
# Ala's "Positive / Appreciation is just a generic AI header" point).
EXCLUDED_THEMES = {"Positive / Appreciation", "Overall Dissatisfaction"}

ACTION_TYPES = ["LOCAL ACTION", "LOCAL PROJECT", "WORKSTREAM / ENTERPRISE", "MONITOR / PROTECT"]

SUPPORT_TEAMS = [
    "Workplace Solutions", "Workplace Experience", "Technology", "Food & Beverage",
    "Workplace Operations", "Change & Communications", "Reservation-system owner",
]

PLANS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "action_plans")


def default_owner() -> str:
    return os.environ.get("PULSEPOINT_ACTION_OWNER", DEFAULT_OWNER)


def _provider() -> str:
    # Default to local Ollama — free, and keeps comments on the machine.
    return (os.environ.get("PULSEPOINT_LLM_PROVIDER") or "ollama").strip().lower()


def _model(provider: str) -> str:
    m = os.environ.get("PULSEPOINT_LLM_MODEL")
    if m:
        return m
    return {
        "ollama": "llama3.1",
        "openai": "gpt-4o",
        "azure": os.environ.get("PULSEPOINT_AZURE_DEPLOYMENT", ""),
        "anthropic": "claude-3-5-sonnet-latest",
    }.get(provider, "")


def configured_provider():
    """Return the provider name if it is usable (keys present), else None.

    Ollama needs no key (local), so it counts as configured by default; cloud
    providers count only when their key/endpoint env vars are set. Used to gate
    the UI so the card degrades gracefully instead of erroring."""
    p = _provider()
    if p == "ollama":
        return "ollama"
    if p == "openai" and (os.environ.get("OPENAI_API_KEY") or os.environ.get("PULSEPOINT_OPENAI_API_KEY")):
        return "openai"
    if p == "azure" and os.environ.get("PULSEPOINT_AZURE_ENDPOINT") and \
            os.environ.get("PULSEPOINT_AZURE_DEPLOYMENT") and os.environ.get("PULSEPOINT_AZURE_API_KEY"):
        return "azure"
    if p == "anthropic" and (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("PULSEPOINT_ANTHROPIC_API_KEY")):
        return "anthropic"
    return None


# ---------------------------------------------------------------------------
# Persistence — generate once, store per site, render from the stored copy
# ---------------------------------------------------------------------------
def _slug(site: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(site)).strip("_").lower()


def plan_path(site: str) -> str:
    return os.path.join(PLANS_DIR, f"{_slug(site)}.json")


def plan_exists(site: str) -> bool:
    return os.path.exists(plan_path(site))


def load_plan(site: str):
    try:
        with open(plan_path(site), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def save_plan(site: str, plan: dict) -> str:
    os.makedirs(PLANS_DIR, exist_ok=True)
    p = plan_path(site)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=2, ensure_ascii=False)
    return p


def plan_meta(site: str):
    plan = load_plan(site)
    if not plan:
        return None
    return plan.get("_meta")


# ---------------------------------------------------------------------------
# Payload — assemble ONLY this site's real evidence for the model
# ---------------------------------------------------------------------------
def _round(x, n=1):
    try:
        if x is None:
            return None
        import math
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return round(float(x), n)
    except Exception:
        return None


def _cat(theme: str) -> str:
    return THEME_TO_CATEGORY.get(theme, "")


def build_report_card_payload(data: dict, include_verbatims: bool = True) -> dict:
    """Compact JSON payload of a site's report card. Pure function of `data`
    (the dict returned by site_report_data). Only real fields; unweighted Index
    only; generic buckets excluded."""
    card = data.get("card") or {}

    def _df_records(df, cols):
        if df is None or not len(df):
            return []
        out = []
        for _, r in df.iterrows():
            out.append({k: r[v] for k, v in cols.items() if v in r})
        return out

    # Dimension scores (0-100) + company average + rank/tier
    dimension_scores = []
    for a in (card.get("areas") or []):
        dimension_scores.append({
            "dimension": a.get("dim"),
            "score_0_100": _round(a.get("score")),
            "company_avg": _round(a.get("org")),
            "rank_of_sites": a.get("rank"),
            "tier": a.get("tier"),
        })

    # Importance vs Support, per dimension (1-5, Gap, status)
    imp_vs_sup = []
    dsum = data.get("dim_summary")
    if dsum is not None and len(dsum):
        for _, r in dsum.iterrows():
            gap = r.get("gap5")
            imp_vs_sup.append({
                "dimension": r.get("dimension"),
                "importance_1_5": _round(r.get("mean_importance")),
                "support_1_5": _round(r.get("mean_support")),
                "gap": (None if gap is None or (isinstance(gap, float) and gap != gap) else _round(gap)),
                "agreement_1_5": _round(r.get("agree")) if "agree" in r else None,
            })

    # Highest-Friction survey items (importance dims) — friction 0-100 + %under-supported.
    # A top-5 plan needs more than the card's 4-item list, so take the top 8 from the
    # full item detail when it's available.
    highest_friction = []
    under = data.get("under")
    det = data.get("detail")
    if det is not None and len(det) and {"mean_importance", "friction"} <= set(det.columns):
        under = det[det["mean_importance"].notna()].sort_values("friction", ascending=False).head(8)
    if under is not None and len(under):
        for _, r in under.iterrows():
            highest_friction.append({
                "item": r.get("item"),
                "dimension": r.get("dimension"),
                "friction_0_100": _round(r.get("friction"), 0),
                "pct_under_supported": _round(r.get("friction_pct"), 0),
            })

    # Computed priority ranking (by comment CATEGORY) — the anchor.
    priority_ranking = []
    P = data.get("P")
    if P is not None and len(P):
        for _, r in P.iterrows():
            priority_ranking.append({
                "category": r.get("key"),
                "comments": int(r.get("volume")) if r.get("volume") == r.get("volume") else None,
                "pct_negative": _round((r.get("neg_share") or 0) * 100, 0),
                "friction_0_100": _round(r.get("friction"), 0),
                "priority_score": _round(r.get("Priority"), 0),
            })

    # Digital & Tech items are agreement-only (1-5, no Importance/Gap): send the score,
    # its band and the company average — never a friction / under-supported figure.
    tech_items = []
    for item, site_v, org_v in (data.get("tech_items") or []):
        if site_v is None or site_v != site_v:  # missing / NaN
            continue
        band = ("Strong" if site_v >= 4.0 else "Moderate" if site_v >= 3.5 else "Needs attention")
        tech_items.append({"item": item, "agreement_1_5": _round(site_v, 2),
                           "company_avg_1_5": _round(org_v, 2), "band": band})

    # Candidate priority THEMES (theme-level, high-negative), generic buckets dropped.
    # Prefer the longer list (up to 10) so the model has enough distinct candidates for
    # five priorities; fall back to the card's 5-item list.
    candidate_themes = []
    for t, cnt, neg in (data.get("fix_candidates") or data.get("fixes") or []):
        if t in EXCLUDED_THEMES:
            continue
        candidate_themes.append({
            "theme": t, "category": _cat(t),
            "comments": int(cnt), "pct_negative": _round(neg * 100, 0),
        })

    # Strengths to protect (low-negative themes), generic buckets dropped.
    strengths = []
    for t, cnt, pos in (data.get("strengths") or []):
        if t in EXCLUDED_THEMES:
            continue
        strengths.append({
            "theme": t, "category": _cat(t),
            "comments": int(cnt), "pct_positive": _round(pos * 100, 0),
        })

    # Benchmark vs company (theme rate)
    bench = []
    for t, site_rate, org_rate in (data.get("bench") or []):
        if t in EXCLUDED_THEMES:
            continue
        bench.append({
            "theme": t, "site_rate_pct": _round(site_rate * 100, 0),
            "company_rate_pct": _round(org_rate * 100, 0),
        })

    payload = {
        "location": data.get("site"),
        "respondents_at_site": data.get("all_participants") or data.get("participation"),
        "comments_analyzed": data.get("n"),
        "overall_index_score_0_100_unweighted": _round(card.get("overall")),
        "overall_index_tier": card.get("tier"),
        "overall_index_rank": (f"{card.get('rank')} of {card.get('n_sites')}"
                               if card.get("rank") else None),
        "company_index_score_0_100": _round(card.get("org_overall")),
        "overall_comment_sentiment": {
            "pct_negative": _round(data.get("neg"), 0),
            "pct_positive": _round(data.get("pos"), 0),
        },
        "dimension_scores": dimension_scores,
        "importance_vs_support": imp_vs_sup,
        "highest_friction_items": highest_friction,
        "digital_tech_agreement_items": tech_items,
        "computed_priority_ranking_by_category": priority_ranking,
        "candidate_priority_themes": candidate_themes,
        "strengths_to_protect": strengths,
        "benchmark_vs_company": bench,
        "reference": {
            "action_types": ACTION_TYPES,
            "support_teams": SUPPORT_TEAMS,
            "note": ("Index & dimension scores are 0-100. Gap = Support - Importance (1-5). "
                     "Friction is 0-100 (higher = worse). The priority ranking is by comment "
                     "category; define priorities at the theme level within the top categories. "
                     "Digital & Tech items are agreement-only (1-5): cite the score, band and "
                     "company average, never a friction or under-supported figure."),
        },
    }

    if include_verbatims:
        # the card's one-per-top-theme quotes, plus up to two per candidate theme so every
        # possible priority has real examples to draw on (deduplicated by text)
        reps, seen = [], set()
        for q in list(data.get("quotes") or []) + list(data.get("cand_quotes") or []):
            txt = q.get("text")
            if q.get("theme") in EXCLUDED_THEMES or not txt or txt in seen:
                continue
            seen.add(txt)
            reps.append({"theme": q.get("theme"), "sentiment": q.get("sentiment"),
                         "role": q.get("role"), "text": txt})
        payload["representative_comments"] = reps
        payload["positive_comments"] = [
            {"theme": q.get("theme"), "role": q.get("role"), "text": q.get("text")}
            for q in (data.get("pos_quotes") or []) if q.get("theme") not in EXCLUDED_THEMES
        ]
    else:
        payload["representative_comments"] = "OMITTED — aggregate-only mode (no verbatim text sent)"
        payload["positive_comments"] = "OMITTED — aggregate-only mode"

    return payload


# ---------------------------------------------------------------------------
# The v3 master prompt (clean — callouts stripped; §1 aligned to the real
# category-level computed ranking).
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """Act as a Workplace Strategy, Corporate Real Estate, Workplace Operations and Change Management expert.

Using ONLY the PulsePoint Experience Index Report Card data provided in the user message (between <report_card> tags), create an "Operational Action Plan — What to Do Next" for the location. That data is the sole source of truth. Do not use any other location's data and do not invent facts. If a field you would normally cite is absent, note it under assumptions instead of fabricating it.

PURPOSE
This is NOT a re-summary of the survey. Translate the findings into a top 5 of clear, location-specific recommendations telling the owner what is happening, what to prioritize, what to do, what evidence supports it, who should support, and what they must do next.

DATA VOCABULARY (use these exact structures; do not invent others)
- Overall PulsePoint Experience Index Score (0-100): the unweighted average of the four Dimension Scores. This is the only Index figure you may cite. Never reference any "weighted" variant.
- Four Dimension Scores (0-100): Work Activity, Physical Feature, Service Level, Digital & Tech. When you cite a "score", it is a Dimension Score, not a category.
- Importance vs. Support items: each has Support and Importance (1-5), a Gap (Support - Importance), and a status. Under-supported % and Friction (0-100) come from here.
- Digital & Tech items are agreement-only (1-5, no Importance): cite the agreement score, its band (Strong >= 4.0, Moderate 3.5-3.9, Needs attention < 3.5) and the company average. Never quote a Gap, Friction or under-supported % for them.
- 7 Comment Categories (groupings) and 29 Themes (the atomic unit). Themes are the level at which you define priorities.
- Computed Priority ranking: the card ranks comment CATEGORIES by 0.40*volume + 0.35*%negative + 0.25*friction. Use this ranking to identify the most material areas; then define priorities at the THEME level within/across the top categories.
- Representative quotes: the only source for concrete examples.
- Ignore the themes "Positive / Appreciation" and "Overall Dissatisfaction" — they are generic and non-actionable (they are already excluded from the data you were given).

EXPECTATIONS
1. Start from the computed Priority ranking and the candidate priority themes provided; do not re-rank from scratch. Weigh frequency + severity + impact + quantitative + qualitative evidence together. Volume is already 40% of the ranking, so do not lean on volume alone.
2. Preserve THEME integrity: do not merge two distinct themes into one priority just because they sound related or share a parent category (e.g. Furniture & Ergonomics vs. Temperature/Thermal are separate even though both sit under Physical Environment). Combine only if this location's own evidence clearly shows one underlying problem. Do not create artificial groupings to fit the five priorities; list extra material issues as secondary/monitor items.
3. Identify the actual workplace problem, not the theme label. Say what is happening, who/what work is affected, and how it shows up day-to-day.
4. Recommend the top 5 priorities: five distinct, material issues with no overlap between them. Return fewer only if this location's evidence does not support five distinct issues, and say why under assumptions; never pad with weak, duplicate or overlapping items.
5. For each priority give: a specific name; the source theme(s); the issue (<= 2 sentences); what it looks like here (<= 3 items, drawn ONLY from the representative quotes provided — never extrapolate; if the quotes lack detail, say local validation is required); evidence (real fields only, each tagged Finding/Interpretation/Recommendation); a specific proposed action (no vague verbs); the proposed team (Primary is always the owner, plus appropriate Support); a direct "site team must do" instruction in action verbs; and one action type.
6. Do not jump straight to a capital solution: prefer utilization / space allocation / booking rules / operations / behavior / technology / maintenance / communication first, and recommend a validation step where the root cause is not established.
7. Action types (choose one): LOCAL ACTION, LOCAL PROJECT, WORKSTREAM / ENTERPRISE, MONITOR / PROTECT. Do NOT use "Quick Win".
8. Protect What Is Working: 2-5 real, specific strengths (from strengths_to_protect); for each say "Maintain the current service level and monitor performance." Never build this from the excluded generic buckets.

QUALITY BAR (apply before answering; do not print)
Concise; logical chain Evidence -> Issue -> Example -> Action -> Team -> Next step; explicit; adaptive to THIS location; source-disciplined (never present interpretation as fact; never fabricate scores, %s, comments, equipment, space, utilization, costs, policies or root causes). The owner for every priority is: {OWNER}.

OUTPUT — return ONE valid JSON object and nothing else, matching exactly this shape:
{{
  "location": "string",
  "indexScore": number,
  "indexTier": "string",
  "reviewFullReportCardRecommended": true,
  "priorities": [
    {{
      "name": "string",
      "sourceThemes": ["string"],
      "issue": "string (<= 2 sentences)",
      "whatItLooksLike": ["string (from supplied quotes only; max 3)"],
      "evidence": [{{"type": "Finding|Interpretation|Recommendation", "text": "string"}}],
      "proposedAction": "string",
      "primaryOwner": "{OWNER}",
      "supportTeams": ["string"],
      "siteMustDo": "string",
      "actionType": "LOCAL ACTION|LOCAL PROJECT|WORKSTREAM / ENTERPRISE|MONITOR / PROTECT"
    }}
  ],
  "protectWhatIsWorking": [{{"area": "string", "evidence": "string", "note": "Maintain the current service level and monitor performance."}}],
  "operationalResponseTable": [{{"priorityName": "string", "validate": "Confirm / Modify / Reject", "actionType": "", "finalOwner": "", "cost": "", "targetOrEscalation": ""}}],
  "assumptions": ["string (only material ones)"],
  "evidenceConfidence": [{{"priorityName": "string", "supportingDataPoints": ["string"], "needsLocalValidation": ["string"]}}]
}}

Rules for the JSON: 5 priorities (fewer only when the evidence does not support five distinct issues), ordered by the computed ranking; one operationalResponseTable row per priority with the decision fields (actionType, finalOwner, cost, targetOrEscalation) left blank; reviewFullReportCardRecommended always true; evidenceConfidence replaces any 1-10 score. Output JSON only — no prose, no code fences."""


def build_prompt(owner: str) -> str:
    # The template escapes literal JSON braces as {{ }} (str.format style); since the
    # owner is filled with replace(), un-escape them too so the model sees valid JSON.
    return PROMPT_TEMPLATE.replace("{OWNER}", owner).replace("{{", "{").replace("}}", "}")


# ---------------------------------------------------------------------------
# LLM backends — all via stdlib urllib (no extra dependencies). temperature 0.
# ---------------------------------------------------------------------------
def _http_json(url: str, payload: dict, headers: dict, timeout: int = 300) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _call_ollama(system: str, user: str, model: str) -> str:
    host = os.environ.get("PULSEPOINT_OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    out = _http_json(f"{host}/api/chat", {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }, headers={})
    return (out.get("message") or {}).get("content", "")


def _call_openai(system: str, user: str, model: str) -> str:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("PULSEPOINT_OPENAI_API_KEY")
    base = os.environ.get("PULSEPOINT_OPENAI_BASE", "https://api.openai.com/v1").rstrip("/")
    out = _http_json(f"{base}/chat/completions", {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }, headers={"Authorization": f"Bearer {key}"})
    return out["choices"][0]["message"]["content"]


def _call_azure(system: str, user: str, model: str) -> str:
    endpoint = os.environ["PULSEPOINT_AZURE_ENDPOINT"].rstrip("/")
    deployment = os.environ["PULSEPOINT_AZURE_DEPLOYMENT"]
    api_version = os.environ.get("PULSEPOINT_AZURE_API_VERSION", "2024-06-01")
    key = os.environ["PULSEPOINT_AZURE_API_KEY"]
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    out = _http_json(url, {
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }, headers={"api-key": key})
    return out["choices"][0]["message"]["content"]


def _call_anthropic(system: str, user: str, model: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("PULSEPOINT_ANTHROPIC_API_KEY")
    out = _http_json("https://api.anthropic.com/v1/messages", {
        "model": model,
        "max_tokens": 4096,
        "temperature": 0,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }, headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
    parts = out.get("content") or []
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text")


_BACKENDS = {"ollama": _call_ollama, "openai": _call_openai,
             "azure": _call_azure, "anthropic": _call_anthropic}


def _extract_json(text: str) -> dict:
    """Robustly pull a JSON object out of a model response."""
    if not text or not text.strip():
        raise ValueError("empty model response")
    t = text.strip()
    # strip code fences if present
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(t[start:end + 1])
    raise ValueError("model did not return valid JSON")


# ---------------------------------------------------------------------------
# Validation — repair minor issues, raise on structural failure
# ---------------------------------------------------------------------------
def _norm_action_type(v: str) -> str:
    s = str(v or "").strip().upper().replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    for at in ACTION_TYPES:
        if s == at:
            return at
    if "MONITOR" in s or "PROTECT" in s:
        return "MONITOR / PROTECT"
    if "ENTERPRISE" in s or "WORKSTREAM" in s:
        return "WORKSTREAM / ENTERPRISE"
    if "PROJECT" in s:
        return "LOCAL PROJECT"
    if "QUICK" in s or "LOCAL" in s:
        return "LOCAL ACTION"
    return "LOCAL ACTION"


def validate_plan(plan: dict, owner: str) -> dict:
    if not isinstance(plan, dict):
        raise ValueError("plan is not a JSON object")
    pri = plan.get("priorities")
    if not isinstance(pri, list) or not pri:
        raise ValueError("plan has no priorities")
    plan["priorities"] = pri[:5]  # the plan targets a top 5 -> keep at most 5
    for p in plan["priorities"]:
        p.setdefault("name", "Priority")
        p.setdefault("issue", "")
        p.setdefault("sourceThemes", [])
        p.setdefault("whatItLooksLike", [])
        p.setdefault("evidence", [])
        p.setdefault("proposedAction", "")
        p["primaryOwner"] = owner
        p.setdefault("supportTeams", [])
        p.setdefault("siteMustDo", "")
        p["actionType"] = _norm_action_type(p.get("actionType"))
        # keep whatItLooksLike to <=3
        if isinstance(p["whatItLooksLike"], list):
            p["whatItLooksLike"] = p["whatItLooksLike"][:3]
    plan.setdefault("location", "")
    plan.setdefault("protectWhatIsWorking", [])
    plan.setdefault("assumptions", [])
    plan.setdefault("evidenceConfidence", [])
    plan["reviewFullReportCardRecommended"] = True
    # (re)build the response table from the final priority set for consistency
    plan["operationalResponseTable"] = [
        {"priorityName": p["name"], "validate": "Confirm / Modify / Reject",
         "actionType": "", "finalOwner": "", "cost": "", "targetOrEscalation": ""}
        for p in plan["priorities"]
    ]
    return plan


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate_action_plan(data: dict, owner: str = None, include_verbatims: bool = True) -> dict:
    """Generate, validate, and return an action-plan dict for one site.
    Raises RuntimeError with an actionable message if no backend is usable, or
    ValueError/urllib errors on a failed call. Does NOT persist — call save_plan."""
    owner = owner or default_owner()
    provider = configured_provider()
    if provider is None:
        raise RuntimeError(
            "No model endpoint configured. Recommended: run a local model with Ollama so "
            "comments never leave the machine — install Ollama, `ollama pull llama3.1`, then set "
            "PULSEPOINT_LLM_PROVIDER=ollama. Or configure Azure/OpenAI/Anthropic (see README).")

    model = _model(provider)
    if not model:
        raise RuntimeError(f"No model name set for provider '{provider}'. "
                           "Set PULSEPOINT_LLM_MODEL (or PULSEPOINT_AZURE_DEPLOYMENT for Azure).")

    payload = build_report_card_payload(data, include_verbatims=include_verbatims)
    system = build_prompt(owner)
    user = f"<report_card>\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n</report_card>"

    t0 = time.time()
    raw = _BACKENDS[provider](system, user, model)
    plan = validate_plan(_extract_json(raw), owner)

    plan["_meta"] = {
        "generated_at": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "provider": provider,
        "model": model,
        "owner": owner,
        "include_verbatims": bool(include_verbatims),
        "elapsed_s": round(time.time() - t0, 1),
        "prompt_version": "v3.1 (top 5)",
    }
    return plan


# ---------------------------------------------------------------------------
# HTML rendering — matches the site card's look & feel (classes added to the
# report <style> block in app.py). Pure string builder; no Streamlit here.
# ---------------------------------------------------------------------------
def _esc(s) -> str:
    import html as _h
    return _h.escape("" if s is None else str(s))


_TAG_CLASS = {
    "LOCAL ACTION": "oap-tag-local",
    "LOCAL PROJECT": "oap-tag-proj",
    "WORKSTREAM / ENTERPRISE": "oap-tag-ent",
    "MONITOR / PROTECT": "oap-tag-mon",
}


def render_action_plan_html(plan: dict, owner: str = None) -> str:
    if not plan or not plan.get("priorities"):
        return ""
    owner = owner or (plan.get("_meta") or {}).get("owner") or default_owner()
    meta = plan.get("_meta") or {}

    blocks = []
    for i, p in enumerate(plan["priorities"], 1):
        at = p.get("actionType", "LOCAL ACTION")
        tag_cls = _TAG_CLASS.get(at, "oap-tag-local")
        looks = "".join(f"<li>{_esc(x)}</li>" for x in (p.get("whatItLooksLike") or []))
        looks_html = f"<div class='oap-look'><span class='oap-lbl'>What this looks like here</span><ul>{looks}</ul></div>" if looks else ""
        ev = "".join(
            f"<li><span class='oap-ev-t'>{_esc(e.get('type', 'Finding'))}</span> {_esc(e.get('text'))}</li>"
            for e in (p.get("evidence") or []) if e.get("text"))
        ev_html = f"<div class='oap-ev'><span class='oap-lbl'>Evidence</span><ul>{ev}</ul></div>" if ev else ""
        support = ", ".join(p.get("supportTeams") or []) or "—"
        blocks.append(
            "<div class='oap-pri'>"
            f"<div class='oap-pri-h'><span class='oap-num'>{i}</span>"
            f"<span class='oap-name'>{_esc(p.get('name'))}</span>"
            f"<span class='oap-tag {tag_cls}'>{_esc(at)}</span></div>"
            f"<div class='oap-issue'>{_esc(p.get('issue'))}</div>"
            f"{looks_html}{ev_html}"
            f"<div class='oap-line'><span class='oap-lbl'>Proposed action</span> {_esc(p.get('proposedAction'))}</div>"
            f"<div class='oap-line'><span class='oap-lbl'>Team</span> Primary: <b>{_esc(owner)}</b> · Support: {_esc(support)}</div>"
            f"<div class='oap-line'><span class='oap-lbl'>Site team must</span> {_esc(p.get('siteMustDo'))}</div>"
            "</div>"
        )

    protect = ""
    if plan.get("protectWhatIsWorking"):
        items = "".join(
            f"<li><b>{_esc(x.get('area'))}</b> — {_esc(x.get('evidence'))} "
            f"<span class='oap-mut'>{_esc(x.get('note') or 'Maintain the current service level and monitor performance.')}</span></li>"
            for x in plan["protectWhatIsWorking"])
        protect = f"<div class='oap-sub2'>Protect what is working</div><ul class='oap-protect'>{items}</ul>"

    # Operational response table (decision fields blank for the team)
    rows = "".join(
        f"<tr><td>{_esc(r.get('priorityName'))}</td><td>{_esc(r.get('validate'))}</td>"
        "<td></td><td></td><td></td><td></td></tr>"
        for r in (plan.get("operationalResponseTable") or []))
    table = ("<div class='oap-sub2'>Operational team response</div>"
             "<table class='oap-tbl'><tr><th>Priority</th><th>Validate</th><th>Action type</th>"
             "<th>Final owner</th><th>Cost</th><th>Target / escalation</th></tr>"
             f"{rows}</table>") if rows else ""

    assumptions = ""
    if plan.get("assumptions"):
        lis = "".join(f"<li>{_esc(a)}</li>" for a in plan["assumptions"])
        assumptions = f"<div class='oap-sub2'>Assumptions</div><ul class='oap-assume'>{lis}</ul>"

    stamp = ""
    if meta:
        stamp = (f"<div class='oap-stamp'>Auto-generated draft · {_esc(meta.get('provider'))}/"
                 f"{_esc(meta.get('model'))} · {_esc(meta.get('generated_at'))}"
                 f"{' · aggregate-only' if meta.get('include_verbatims') is False else ''}</div>")

    return (
        "<section class='oap'>"
        "<div class='oap-hd'>"
        "<div class='oap-t'>Operational Action Plan — What to Do Next</div>"
        "<div class='oap-note'>Auto-generated draft to validate — not a re-summary of the survey. "
        f"Primary owner for every action: <b>{_esc(owner)}</b>. A full read of the report card below "
        "is still recommended. The team's job: Confirm / Modify / Reject, then assign owner, cost, and timing.</div>"
        "</div>"
        f"{''.join(blocks)}{protect}{table}{assumptions}{stamp}"
        "</section>"
    )


# CSS injected into the report <style> block (kept here so markup + styles live together)
ACTION_PLAN_CSS = """
 .oap{border:1px solid #d6dcea;border-radius:10px;background:#f7f9fd;padding:16px 18px;margin:14px 0 22px}
 .oap-hd{border-bottom:1px solid #e2e7f1;padding-bottom:9px;margin-bottom:11px}
 .oap-t{font-size:17px;font-weight:800;color:#1f2d5a;letter-spacing:.01em}
 .oap-note{font-size:11.5px;color:#5b6577;margin-top:4px;line-height:1.5}
 .oap-pri{background:#fff;border:1px solid #e6e9f0;border-radius:8px;padding:11px 13px;margin:9px 0}
 .oap-pri-h{display:flex;align-items:center;gap:9px;flex-wrap:wrap}
 .oap-num{background:#1f2d5a;color:#fff;font-weight:700;font-size:12px;width:22px;height:22px;border-radius:50%;
   display:inline-flex;align-items:center;justify-content:center;flex:none}
 .oap-name{font-size:14.5px;font-weight:700;color:#1f2d5a}
 .oap-tag{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;padding:3px 8px;border-radius:20px;margin-left:auto}
 .oap-tag-local{background:#e6f0ff;color:#1c5bb8} .oap-tag-proj{background:#fff0d9;color:#a5600a}
 .oap-tag-ent{background:#efe3fb;color:#6b3fa0} .oap-tag-mon{background:#e2f4e8;color:#1f7a44}
 .oap-issue{font-size:13px;color:#26303f;margin:7px 0 4px;line-height:1.5}
 .oap-lbl{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;color:#8a93a6;margin-right:6px}
 .oap-look,.oap-ev{margin:5px 0}
 .oap-look ul,.oap-ev ul{margin:3px 0 3px 0;padding-left:19px}
 .oap-look li,.oap-ev li{font-size:12px;color:#3a4453;margin:2px 0;line-height:1.45}
 .oap-ev-t{font-size:9.5px;font-weight:700;text-transform:uppercase;color:#1f7a44;background:#e8f5ee;padding:1px 5px;border-radius:4px;margin-right:5px}
 .oap-line{font-size:12.5px;color:#26303f;margin:5px 0;line-height:1.5}
 .oap-sub2{font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:.05em;color:#1f2d5a;margin:14px 0 5px}
 ul.oap-protect,ul.oap-assume{margin:3px 0;padding-left:19px}
 ul.oap-protect li,ul.oap-assume li{font-size:12px;color:#3a4453;margin:3px 0;line-height:1.45}
 .oap-mut{color:#8a93a6;font-style:italic}
 table.oap-tbl{width:100%;border-collapse:collapse;margin-top:5px;font-size:11.5px}
 table.oap-tbl th{background:#eef2f9;color:#3a4453;text-align:left;padding:5px 7px;border:1px solid #dde3ee;font-size:10.5px;text-transform:uppercase;letter-spacing:.03em}
 table.oap-tbl td{padding:5px 7px;border:1px solid #e6e9f0;min-height:20px}
 .oap-stamp{font-size:10px;color:#9aa0ae;margin-top:11px;border-top:1px solid #e8ecf3;padding-top:7px}
"""
