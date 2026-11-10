# PulsePoint — Open-Ended Response Analyzer

An interactive Streamlit tool for reviewing and analyzing the open-ended comments
from the PulsePoint workplace survey. It links the *why* (comments) to the *what*
(the importance-vs-support gaps), surfaces themes and sentiment, and lets you
review and correct the auto-tagging.

## What's in the box

```
pulsepoint_tool/
├── app.py              # the Streamlit app
├── build_data.py       # regenerates the processed data from a raw export
├── taxonomy.py         # the theme taxonomy + junk detection (edit to tune tags)
├── sentiment.py        # domain-tuned ensemble sentiment model (edit to tune)
├── requirements.txt    # dependencies
├── data/
│   ├── raw.xlsx               # your raw survey export
│   ├── comments.csv           # one row per comment, auto-tagged (theme, sentiment, junk)
│   ├── scores_overall.csv     # Tableau-spec Friction/Composite/RYG per item (completers, 10 sites)
│   ├── scores_by_site.csv     # same, per in-scope site
│   ├── scores_dimension.csv   # dimension composites + weighted/equal overall
│   ├── scores_*_all.csv       # same three, all-responses population (for the toggle)
│   ├── gap_overall.csv        # legacy signed gap (retained; app now uses Friction)
│   ├── gap_by_site.csv        # legacy signed gap by site
│   ├── respondent_scores.csv  # per-respondent overall-agreement scores (key drivers)
│   ├── emerging_themes.json   # auto-discovered clusters of Unclassified comments
│   └── meta.json              # filter option lists
└── README.md
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
single **priority score** that combines comment volume, share negative, and the
importance-vs-support gap, so you see *where to act first* instead of eyeballing three
charts. Colored by % negative, with a table of the components and 2–3 representative
negative quotes per item. Filter to one site in the sidebar to get that site's own
priority list (the gap becomes site-specific too). Score = 40% volume + 35% share
negative + 25% importance-gap, each rescaled 0–1.

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

**📄 Site Reports** — generate a leadership-ready one-pager for any site, laid out in seven
sections. **(1) Overall PulsePoint Experience Index Score** — the 0–100 headline, computed as the
**unweighted** average of the four Dimension Scores so it **matches the Tableau deck exactly**;
shown with the site's **rank among the 10 in-scope sites** and its **tier** (top/middle/bottom
third of the distribution — a relative standing, not an arbitrary A–F cutoff). **(2) Dimension
Scores** — the four dimension composites (0–100) with per-dimension rank. **(3) Importance vs.
Support gaps** — the summary and by-question detail (1–5), including **Digital & Tech** with a real
status band (Strong ≥4.0 / Moderate 3.5–3.9 / Needs attention <3.5, since it is agreement-only).
**(4) Associate Sentiment Score** — a standalone section for the survey's summary sentiment items
(Q8/Q12/Q16/Q19; Q16 = Services, Q19 = Technology by survey position), each with a **Welch t-test**
verdict vs. the other in-scope sites. **(5) Friction & strength themes** — a large red Friction chart
and green Strengths chart, dimension-labeled, with wrapped question text and the score + % on each
bar. **(6) Open-ended comment themes** — the comment-sentiment headline, sentiment and theme charts
(theme bars labeled `% | N comments`), top priorities, a what's-working vs what-to-fix split, a
benchmark vs the org, representative quotes, a categorization methodology note, and the full
comments-by-category-and-sentiment appendix. **(7) Demographics — respondent profile** — a visual
rollup (role, tenure, hybrid category) above a detailed backup table. A **data source & method**
footer documents the pipeline. Each card can also open with an optional, auto-generated
**Operational Action Plan — "What to Do Next"** band (see its own section below). Download per site as
**HTML, PDF, or PowerPoint**, build a full deck of all sites at once, or export a site's complete
verbatim bank as CSV.

At the top of the tab is the **10 PulsePoints — Portfolio Report Card**: a company-wide exec brief
(2–3 pages) that rolls the same elements up across all ten sites — the company Index Score, a ranked
**site leaderboard**, a **site × dimension heatmap**, company Importance-vs-Support gaps, company-wide
Friction and strength themes, the top comment themes with a **site × theme heatmap**, and the overall
respondent profile. Download as HTML or PDF.

**✏️ Tagging Workspace** — starts with a **"Discover missing themes"** panel that
auto-clusters the Unclassified comments so you can spot themes the taxonomy is missing
(it already surfaced an "office supplies" cluster, for example). Below that, review the
auto-tags and override them. Edit the
`sentiment_override`, `themes_override` (pipe-separated, e.g. `Privacy & Confidentiality|Noise & Distraction`),
and `junk_override` columns, add notes, tick *reviewed*, then **Save changes**.
Saves to `data/overrides.csv` and updates every tab. Export the fully-tagged
dataset (with your edits applied) to CSV or Excel.

## Operational Action Plan — "What to Do Next"

Each site card carries an auto-generated **Operational Action Plan**, placed just below the *Vs. org —
theme rates* table — the bridge between the survey read-out and execution, per the work-stream review. It
is **not** a re-summary: it converts the site's own evidence into a **top 5 of prioritized,
validate-before-acting recommendations** (five distinct issues, ordered by the card's computed category
ranking; fewer only if the evidence doesn't support five), each with the
specific issue, what it looks like at that site (quoted from real comments), the supporting evidence, a
proposed action, the proposed support team, and a required next step — plus a "Protect what's working"
block and a blank operational-response table for the team to complete (Confirm/Modify/Reject → owner →
cost → timing). The primary owner for every action defaults to **Regional Director + Local Team** and is
editable (the ownership model is still an open decision).

The plan is generated **once and stored** per site (`data/action_plans/<site>.json`), so the text is
stable — it does not shift each time the card is opened. Regenerate any time from the **Site Reports**
tab (*🧭 Operational Action Plan — generate / regenerate*), per site or for all ten at once. The section
is grounded strictly in that site's report card (unweighted Index only; the generic "Positive /
Appreciation" and "Overall Dissatisfaction" buckets are excluded), and the model is told to use only the
supplied quotes for examples and never to fabricate. To give it enough to work with for five priorities,
the model receives up to 10 candidate themes (not just the card's top 5), up to two real example quotes
per candidate theme, the eight highest-Friction survey items, and the Digital & Tech items as agreement
scores against the company average (never as Friction, which isn't comparable for those items).

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
| OpenAI | `openai` | `OPENAI_API_KEY` (opt. `PULSEPOINT_LLM_MODEL`, default `gpt-4o`) |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` (opt. `PULSEPOINT_LLM_MODEL`, default `claude-3-5-sonnet-latest`) |

Other knobs: `PULSEPOINT_ACTION_OWNER` sets the default owner label; the **Include verbatim quotes**
toggle can be turned off to send the model **aggregate numbers only** (no comment text) if policy
requires. Requests run at **temperature 0**. If no backend is configured, the card renders without the
section and the tab shows a one-line setup hint — nothing breaks.

> **Hosted providers:** sending comments to a cloud API means employee text leaves your environment —
> use an organization-approved endpoint, or stick with local Ollama. Either way the generated plan is a
> **draft for the site team to validate**, not a final decision; that guardrail is built into the prompt.

## How the auto-tagging works

- **Junk / no-comment** answers (`"no"`, `"n/a"`, `"nothing at this time"`, etc.)
  are flagged so they don't pollute the trends. Toggle "Include junk" in the sidebar
  to see them.
- **Themes** are assigned by a transparent keyword taxonomy in `taxonomy.py`
  (29 themes across 7 categories). Each comment can carry multiple themes — this is
  the "split multi-topic comments" behavior. Granular details roll up into the parent
  theme (e.g. a specific broken elevator → *Building Operations*).
- **Sentiment** uses a **domain-tuned ensemble** built for this survey (in
  `sentiment.py`), not plain VADER. It combines VADER's general polarity with a
  facilities-specific lexicon, clause-aware negation, intensifier handling, a light
  question-framing prior (Q13 asks what "limits" you, so it leans negative), and
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

- To **tune the themes** (add keywords, add a theme, adjust junk rules) or the
  **sentiment lexicon** (add facilities terms to `sentiment.py`), edit the file and
  rebuild:

  ```
  python build_data.py data/raw.xlsx
  ```

  ⚠️ Rebuilding regenerates `comments.csv`. Your manual edits live separately in
  `data/overrides.csv` and are re-applied on top, so they survive a rebuild as long
  as `comment_id`s stay stable (they will if you use the same raw file).

- To load a **new survey export**, drop it in `data/` and run
  `python build_data.py data/your_new_file.xlsx`.
