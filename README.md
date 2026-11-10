# PulsePoint — Open-Ended Response Analyzer

An interactive Streamlit tool for reviewing and analyzing the open-ended comments
from the PulsePoint workplace survey. It links the *why* (comments) to the *what*
(the importance-vs-support gaps), surfaces themes and sentiment, and lets you
review and correct the auto-tagging.

## What's in the box

```
PPindex/
├── app.py              # the Streamlit UI (tabs, filters, cache wrappers)
├── reports.py          # data loading, computation and the site / portfolio / PowerPoint builders
│                       #   (importable without Streamlit: reports.load_context(), build_site_report(), ...)
├── design.py           # the one colour system (CSS variables), contrast helpers, base stylesheet
├── build_data.py       # regenerates everything in data/ from the de-identified raw export
├── taxonomy.py         # theme keyword rules + junk detection + GENERIC_THEMES
├── sentiment.py        # domain-tuned sentiment model (VADER + facilities lexicon + question prior)
├── action_plan.py      # Operational Action Plan: payload, prompt v3.2, grounding check, rendering
├── llm_backends.py     # shared model backends (Ollama default; hosted providers opt-in)
├── llm_tagger.py       # optional model-based comment tagger — OFF unless PULSEPOINT_TAGGER=llm
├── deid.py             # the identifying columns dropped from the raw export
├── waves.py            # survey rounds: data folder, site list, benchmark wording, round 2 site names
├── combined_report.py  # combined portfolio card (round 1 + round 2 side by side)
├── tools/              # deidentify_raw.py, make_gold_sample.py, eval_tagging.py, refresh_plan_evidence.py
├── tests/              # pytest — baseline, de-identification guard, card checks, contrast, rules
├── data/
│   ├── raw_deid.xlsx          # de-identified survey export (tracked); the original raw.xlsx is NOT tracked
│   ├── comments.csv           # one row per comment, auto-tagged (theme, sentiment, junk)
│   ├── scores_overall.csv     # Tableau-spec Friction / Composite / status per item (+ paired-gap SE & 95% CI)
│   ├── scores_by_site.csv     # same, per in-scope site
│   ├── scores_dimension.csv   # dimension composites + weighted/equal overall
│   ├── site_intervals.csv     # 95% intervals for each site's Index and Dimension Scores (bootstrap)
│   ├── *_all.csv              # the same files for the all-responses population (for the toggle)
│   ├── respondent_scores.csv  # per-respondent overall-agreement scores (sentiment table, key drivers)
│   ├── gold/                  # gold_labels.csv (IDs + labels only) and eval_report.md
│   ├── action_plans/          # stored Operational Action Plans, one JSON per site
│   ├── emerging_themes.json   # clusters of Unclassified comments
│   ├── item_support.csv       # respondent × item Support (0–100, no text) for like-for-like round comparisons
│   ├── meta.json              # filter option lists, wave, items not asked
│   └── round2/                # the same files for Round 2 (30 sites), built with --wave round2
├── docs/CHANGES-2026-11.md    # what the November 2026 review pass changed, with before/after
├── docs/ROUND2-2026-11.md     # Round 2: decisions, data findings, results
└── samples/                   # example site and portfolio cards (HTML + PDF)
```

## Running it (first time)

1. Install Python 3.9+ if you don't have it (https://www.python.org/downloads/).
2. Open a terminal in this folder and install dependencies:

   ```
   pip install -r requirements.txt
   ```

   For **PDF export** (the PDF is a faithful render of the HTML report), install the
   Chromium engine once — it's self-contained and needs no system libraries:

   ```
   python -m playwright install chromium
   ```

   You don't strictly need it: every report has an **HTML** download, and your browser's
   **Print → Save as PDF** produces the identical file (it's the same engine). PDF export
   also has a WeasyPrint fallback, but on macOS WeasyPrint needs extra system libraries
   (`brew install pango`), so Chromium is the easy path. If no renderer is available the
   app just hides the PDF button and points you here — it won't error.

3. Launch the app:

   ```
   streamlit run app.py
   ```

   It opens in your browser at `http://localhost:8501`.

That's it — the processed data is already built, so it loads instantly.

## Completion filter

A **"Completed responses only"** toggle sits at the top of the sidebar, **on by default**.
It restricts the entire tool to the 3,037 respondents who reached the end of the survey
(of 3,832 total) — the 795 drop-offs took a median of 84 seconds and answered almost
nothing. Turn it off to include everyone for comparison. It re-bases every number,
including the importance-vs-support gaps (computed for both populations).

## The eight tabs

**📊 Overview** — headline counts, sentiment mix, comments by question and by site.
Everything responds to the sidebar filters.

**🎯 Action Priorities** — the synthesis view. Ranks every theme (or category) by a
single **priority score** that combines how many **people** raised it, the share of its
comments that are negative, and the Tableau-spec Friction score. Filter to one site in the
sidebar to get that site's own list (Friction becomes site-specific too). Score = 40% people
+ 35% share of comments negative + 25% Friction, each rescaled 0–1. Volume counts people,
not comments — one person can answer five open questions — and % negative is computed on
unique comments. The generic buckets "Positive / Appreciation" and "Overall Dissatisfaction"
are excluded from every ranking (`taxonomy.GENERIC_THEMES`).

**⚖️ Scores & Friction** — matches the official **Tableau dashboard methodology**
exactly. **Composite** = average Support on a 0–100 scale (Importance not factored).
**Friction** = each respondent's shortfall of Support vs Importance, floored at 0, then
averaged — so over-supported areas can't mask under-supported ones. Items are colored by
the **Red/Yellow/Green** status (Under Supported / Target / Over Supported) and plotted
on the **Importance-vs-Support quadrant**. Dimension composites roll up to a weighted
**Overall** score (Work 35 · Service 25 · Physical 20 · Digital & Tech 20). Digital &
Tech uses Agreement only, so its Friction = 100 − Support. Pick any item to pull up the
comments whose themes explain its Friction. **Scope: completed responses across the 10
in-scope sites**, per the spec — so the numbers reconcile with the Tableau report.

**🏷️ Theme & Sentiment Explorer** — three controls drive everything: **View by**
Categories (the 7 roll-ups) or Themes (the 29 granular tags); **Break down by** Site,
Role, Tenure, or Hybrid category; and **Heatmap metric** — *% of group* (rate-
normalized, so a 30-response site is comparable to a 900-response one) or raw Counts.
Plus theme/category frequency, sentiment mix, and a searchable comment table. Multi-
topic comments count under *each* theme they mention, so "cold + no privacy" shows up
under both Temperature and Privacy.

**🔎 Browse & Pivot** — three sub-tabs. *Comment browser* shows every comment color-coded
by sentiment with theme tags. *Pivot builder* cross-tabs any two dimensions (theme × role,
sentiment × site, …) with count, %-of-row/column, or average sentiment. *Exports* offers
two take-it-with-you formats: a multi-sheet **Excel workbook** (comments, theme summary,
site×theme and sentiment×theme pivots, a full verbatim bank, and gaps), and a **no-install
HTML dashboard** — a single self-contained file with filters, live theme bars, and a
searchable comment browser that you double-click to open in any browser, no Python needed.

**🔥 Hotspots & Drivers** — two org-wide diagnostics (they use all responses and ignore
the sidebar filters). *Hotspots* flags site/role/tenure groups whose mention rate for a
theme is significantly above the rest of the org, using a two-proportion z-test (p<0.05)
so small groups don't create false alarms — it answers "where is this *unusually* bad?"
*Key drivers* uses the survey's own overall-agreement scores (1–5) to show which themes
are associated with lower or higher agreement: for each theme, the average score of
commenters who raised it vs **other commenters**, with a 95% interval. People who wrote
nothing are left out — as a group they score higher than commenters (4.08 vs 3.57), which
made almost every theme look like a drag. It is an association, not evidence that fixing a
theme would raise the score.

**📄 Site Reports** — a leadership-ready report card for any in-scope site, in this order:

1. **Header** — the Overall PulsePoint Experience Index as "68 ± 1" (95% margin from resampling
   the site's respondents), its **standing** (Above / In line with / Below the company average),
   the company value, respondents and people who commented. The Index is the **unweighted**
   average of the four Dimension Scores, so it **matches the Tableau deck exactly**.
2. **At a glance** (fits page 1 with the header) — the four Dimension Scores with margin and
   standing; the three highest-Friction items with % under-supported and severity; the three
   best-supported items; the top three action-plan priorities; one line each for role, tenure
   and hybrid mix.
3. **Importance vs. Support** — per dimension: score ± margin, standing, "k of n items need
   attention", worst item; then every question with its official status and a severity word.
4. **Associate sentiment** — the Q8/Q12/Q16/Q19 summary items: site mean ± margin, company
   mean, difference, standing.
5. **Top priorities — survey items** — the five highest-Friction items, each with its linked
   comment themes (in people) and one illustrative comment; a **Technology watch** line for
   Digital & Tech items under 3.5 agreement or below the company; a secondary "What comments
   add" table by category.
6. **What needs attention / What's working** — themes by people; "% positive" is the true
   positive share (a theme needs ≥10 comments, ≥50% positive and more positive than negative;
   otherwise the list is filled from best-supported survey items, labelled as ratings).
7. **Themes that differ here** — only themes whose share of people differs from the other nine
   sites beyond chance (two-proportion test, Benjamini–Hochberg across themes).
8. **Operational Action Plan** (see below). 9. **In their words** — representative quotes.
10. **All comments by category** — in the HTML only; print and PDF show a one-line pointer.
11. **Respondent profile** — counts under 5 print as "<5" with the percentage blank.
12. **Method note**, including whether comment tagging has been validated.

**Standing, not rank.** Cards never rank sites or sort them into thirds. A score is Above or
Below the company average only when the printed company value lies outside the site's 95%
interval; otherwise it is In line. "Company" always means the 10 in-scope sites pooled, in the
selected population. **Severity** (slight / moderate / severe from the rounded gap; *borderline*
when the gap's 95% interval includes 0; "On track, low support" for Target items under 60/100)
is added next to the official status and never replaces it. **Attribution:** a comment shows
the writer's role only when at least 10 respondents at the site share it; tenure is never shown
beside a comment. Download per site as **HTML, PDF or PowerPoint** (the slide carries the same
numbers and labels as the card), build a deck of all sites, or export a site's verbatims (CSV —
internal analysis only).

At the top of the tab is the **10 PulsePoints — Portfolio Report Card**: the company Index with
its margin, each site's Index ± margin and standing (alphabetical — no ranking), Dimension Scores
by site, company Importance vs. Support (score ± margin, items needing attention, worst item),
company Friction and strengths, comment themes by people with a site × theme table, the
respondent profile, and a question × site grid with **one value per cell** (Friction for items
with an Importance rating, agreement for Digital & Tech — not comparable), coloured by status
with an icon, other values on hover, printed on a landscape page.

All cards use one colour system (`design.py`): red / amber / green mean support status only,
score badges are neutral, sentiment has its own blue / purple pair, every status carries a word
or icon, and all text meets WCAG AA contrast (tested).

**✏️ Tagging Workspace** — starts with a **"Discover missing themes"** panel that
auto-clusters the Unclassified comments so you can spot themes the taxonomy is missing
(it already surfaced an "office supplies" cluster, for example). Below that, review the
auto-tags and override them. Edit the
`sentiment_override`, `themes_override` (pipe-separated, e.g. `Privacy & Confidentiality|Noise & Distraction`),
and `junk_override` columns, add notes, tick *reviewed*, then **Save changes**.
Saves to `data/overrides.csv` and updates every tab. Export the fully-tagged
dataset (with your edits applied) to CSV or Excel. The **Gold set** panel lists the 300
comments in `data/gold/gold_labels.csv` so a person can confirm or correct each label and tick
*verified*; only verified rows count as evidence of accuracy.

## Operational Action Plan — "What to Do Next"

Each site card carries an auto-generated **Operational Action Plan** after the theme tables and before
the quotes — the bridge between the survey read-out and execution. It is **not** a re-summary: it
converts the site's own evidence into a **top 5 of validate-before-acting recommendations**, each with
the specific issue, what it looks like at that site (quoted from real comments), the supporting
evidence, a proposed action, the proposed support team, and a required next step — plus "Protect what's
working" and a blank operational-response table (Confirm/Modify/Reject → owner → cost → timing). The
primary owner defaults to **Regional Director + Local Team** and is editable.

Plans are generated **once and stored** per site (`data/action_plans/<site>.json`). **Prompt v3.2**
starts priorities from the site's highest-Friction survey items, with comment themes (counted in people)
as supporting evidence, and forbids rank or "lowest/highest of all sites" wording unless the standing is
Above or Below. **v3.2 is a draft that needs work-stream sign-off before anyone regenerates plans.**

**Grounding check.** After generation every number in each evidence line is checked against the data
sent to the model (or a supplied quote), and every "what it looks like" line against the supplied quotes
(normalised; "…" may join fragments). Failures are shown in the Site Reports tab and the plan is **not
saved**.

**The stored plans were not regenerated in November 2026.** `tools/refresh_plan_evidence.py` updated the
numbers inside their evidence lines to the corrected data, replaced the "% positive" protect lines (they
were 100 − % negative), removed rank wording, and flagged the priorities the corrected evidence no longer
clearly supports (shown on each card). Every edit: `docs/plan-evidence-refresh-2026-11.md`.

### Configuring the model (privacy-first)

Generation calls a language model over plain HTTP — **no extra Python packages**. The **default and
recommended** backend is a **local [Ollama](https://ollama.com) model**, so **employee comments never
leave the machine**:

```bash
# one-time
brew install ollama            # or download from ollama.com
ollama pull llama3.1           # any capable local model works
# then just run the app — Ollama is the default provider
```

To use a hosted model instead, set environment variables before launching Streamlit:

| Provider | `PULSEPOINT_LLM_PROVIDER` | Required env vars |
|----------|---------------------------|-------------------|
| Ollama (local, default) | `ollama` | *(none)* — optional `PULSEPOINT_OLLAMA_HOST`, `PULSEPOINT_LLM_MODEL` |
| Azure OpenAI | `azure` | `PULSEPOINT_AZURE_ENDPOINT`, `PULSEPOINT_AZURE_DEPLOYMENT`, `PULSEPOINT_AZURE_API_KEY` (opt. `PULSEPOINT_AZURE_API_VERSION`) |
| OpenAI | `openai` | `OPENAI_API_KEY` (opt. `PULSEPOINT_LLM_MODEL`, default `gpt-6.1-sol`) |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` (opt. `PULSEPOINT_LLM_MODEL`, default `claude-sonnet-5-5`) |

Defaults were checked against each vendor's model documentation in November 2026. Hosted calls have not
been exercised from this repository; newer reasoning models may reject `temperature: 0`, so test once.

Other knobs: `PULSEPOINT_ACTION_OWNER` sets the default owner label; the **Include verbatim quotes**
toggle can be turned off to send the model **aggregate numbers only** (no comment text).
**A hosted provider never receives comment text unless `PULSEPOINT_ALLOW_HOSTED_COMMENTS=1` is set** —
without it, a hosted request with quotes is refused before anything is sent (`llm_backends.py`), and the
toggle defaults to aggregate-only. Requests run at **temperature 0**. If no backend is configured, the card renders without the
section and the tab shows a one-line setup hint — nothing breaks.

> **Hosted providers:** sending comments to a cloud API means employee text leaves your environment —
> use an organization-approved endpoint, or stick with local Ollama. Either way the generated plan is a
> **draft for the site team to validate**, not a final decision; that guardrail is built into the prompt.

## How the auto-tagging works

- **Junk / no-comment** answers (`"no"`, `"n/a"`, `"nothing at this time"`, etc.)
  are flagged so they don't pollute the trends. Toggle "Include junk" in the sidebar
  to see them.
- **Themes** are assigned by a transparent keyword taxonomy in `taxonomy.py`
  (29 themes across 7 categories). Each comment can carry multiple themes. In November
  2026 the rules were tightened so a bare word no longer tags a theme on its own: Desk
  Availability needs a booking / availability / assignment sense (a help desk or "my desk"
  is not enough), AV needs an audio / video / room-technology sense (Microsoft Teams the app
  is not AV), Collaboration Space needs a physical space ("nobody I collaborate with is here"
  goes to In-Person / Team Presence), and substring bugs (Indianapolis → "nap", industry →
  "dust", data entry, sign in, …) are fixed. Every false-positive family has a test with
  invented sentences (`tests/test_taxonomy.py`). The Unclassified share rose from 9.4% to
  12.3%; no catch-all patterns were added to hide it. Precision before and after each rule
  change is in `docs/CHANGES-2026-11.md`.
- **Optional model tagger.** `llm_tagger.py` labels comments with a local model (themes from
  the 29 only, sentiment, and intent) when `PULSEPOINT_TAGGER=llm` is set at build time. It is
  **off** and should stay off unless it beats the rules on verified gold labels
  (`python llm_tagger.py --gold --export …` then `tools/eval_tagging.py --pred llm=…`).
- **Sentiment** uses a **domain-tuned ensemble** built for this survey (in
  `sentiment.py`), not plain VADER. It combines VADER's general polarity with a
  facilities-specific lexicon, clause-aware negation, intensifier handling, a light
  question-framing prior (the "what limits you" and "work types not supported" questions
  lean negative — the prior is keyed by question label; before November 2026 it never fired), and
  explicit *mixed* detection (a comment that's positive AND negative — "great office
  but freezing" — is flagged Neutral / Mixed). Crucially it recognizes **improvement
  requests** ("better monitor stands", "more quiet spaces", "would be helpful if…")
  as unmet needs rather than praise — the single biggest thing generic sentiment gets
  wrong on this data.

  **Accuracy is not yet established.** An earlier claim of "~60% accurate vs ~31% for
  plain VADER" could not be reproduced and has been removed. A 300-comment gold set now
  lives in `data/gold/gold_labels.csv` (IDs and labels only); its rows were pre-filled by an
  AI assistant and must be confirmed by a person in the Tagging Workspace (*Gold set*).
  `python tools/eval_tagging.py` writes `data/gold/eval_report.md` with per-theme
  precision/recall/F1 and sentiment accuracy, separately for verified and draft labels.
  Until at least 30 rows are verified, every card's method note says the comment themes
  and sentiment are automated and not yet validated. Against the draft labels the known
  bias is clear: nearly every sentiment error is *more positive* than the comment, so
  read sentiment shares as a floor on negativity. Lean on theme + Friction context
  alongside sentiment, not sentiment alone.

  *(We evaluated a full RoBERTa transformer too, but chose this offline model so the
  tool needs no ~500MB model download or GPU/torch install and runs anywhere.)*

## Re-tagging or updating the data

- The tracked raw file is the **de-identified** export `data/raw_deid.xlsx`. The original export
  (with network IDs, IPs and locations) stays on disk only and is in `.gitignore`. To refresh it:

  ```
  python tools/deidentify_raw.py data/raw.xlsx data/raw_deid.xlsx
  ```

  `build_data.py` drops those columns on load whatever file it is given, and a test fails if any
  tracked file under `data/` carries one of them.

- To **tune the themes** or the **sentiment lexicon**, edit `taxonomy.py` / `sentiment.py`, bump
  `TAXONOMY_VERSION` if the rules changed, and rebuild:

  ```
  python build_data.py            # reads data/raw_deid.xlsx
  pytest                          # the section-4 baseline must not move
  ```

  ⚠️ Rebuilding regenerates `comments.csv`. Manual edits live in `data/overrides.csv` and are
  re-applied on top as long as `comment_id`s stay stable (same raw file, same row order).

- To load a **new survey export**, de-identify it first, then `python build_data.py data/your_file_deid.xlsx`.

## Survey rounds

Round 1 (10 sites) and Round 2 (30 different sites, same questionnaire) are built and reported
separately; each site's standing is against its own round's average.

```
python tools/deidentify_raw.py ~/path/round2_export.xlsx data/round2/raw_deid.xlsx
python build_data.py --wave round2      # writes data/round2/
python build_data.py                    # round 1, unchanged
```

In the app, the sidebar **Survey round** switch selects the round. The Site Reports tab has the
portfolio card for the selected round and, when both rounds are built, the **combined** card
(both rounds side by side, compared only on items both rounds asked). Round 2 site cards flag
sites under 30 responses as "small sample". Round 2 action plans are stored in
`data/round2/action_plans/`. See `docs/ROUND2-2026-11.md`.

## Privacy and sharing

- Leadership cards apply the attribution rule (role only for groups of 10 or more at the site, never
  tenure). The **offline dashboard, CSV and Excel exports keep role and tenure** and are marked
  **internal analysis only** — do not share them outside the analysis team.
- Comment text itself is not redacted: names that appear inside comments still appear in quotes,
  the comment list and the exports.
- Comment text goes to a hosted model only with `PULSEPOINT_ALLOW_HOSTED_COMMENTS=1`.

## Tests

```
pip install -r requirements-dev.txt
pytest
```

Covers the section-4 baseline (Index by site, company Index, dimension composites, 76/148/11 status
split), the de-identification guard, card content rules (standing, people counts, attribution, "<5",
print-only sections, no rank wording), colour contrast and minimum sizes, the keyword-rule regressions,
sentiment prior, intervals, the gold-set harness, and the model-backend privacy gate (all model calls
stubbed).
