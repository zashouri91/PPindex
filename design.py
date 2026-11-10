"""
Shared visual design for the PulsePoint report cards (site card, portfolio card,
action-plan section). No dependencies — safe to import anywhere.

Rules (Nov 2026 review, phase 7):
  * One meaning per colour. Red / amber / green are reserved for SUPPORT STATUS
    (Under Supported / Target / Over Supported). Score badges are neutral navy;
    standing is a text label. Sentiment has its own blue / purple pair.
  * Never colour alone. Every status carries a word; heatmaps print their numbers.
  * Contrast. Every text colour meets WCAG AA (4.5:1) on the backgrounds it is
    used on — tests/test_design.py reads CONTRAST_PAIRS and fails below AA.
  * Sizes. Notes are at least 11px on screen; dense print tables at least 9px.
"""
import re

# ---------------------------------------------------------------------------
# Colour variables (single source; rendered into :root by css_vars())
# ---------------------------------------------------------------------------
COLORS = {
    # text + surfaces
    "--ink": "#1a1a2e",          # body text
    "--ink-2": "#33415c",        # secondary text
    "--muted": "#55607a",        # notes, captions, legends
    "--navy": "#1f2d5a",         # headings, neutral score badges
    "--navy-2": "#2b3a67",
    "--bg": "#ffffff",
    "--panel": "#f5f7fb",
    "--panel-2": "#eef1f7",
    "--rule": "#dfe3eb",
    "--on-navy": "#ffffff",
    # neutral data bars (no meaning beyond magnitude)
    "--bar": "#4a5a86",
    "--bar-track": "#e6e9f1",
    # SUPPORT STATUS — the only use of red / amber / green
    "--st-under": "#a61b1b",     # Under Supported (text)
    "--st-under-bg": "#fbe3e1",
    "--st-target": "#1d6630",    # Target (text)
    "--st-target-bg": "#e2f1e6",
    "--st-over": "#7a4300",      # Over Supported (text, dark amber)
    "--st-over-bg": "#fcecd2",
    "--st-na": "#4f5a70",        # agreement-only / no status
    "--st-na-bg": "#eceff4",
    # SENTIMENT — its own pair, not red/green
    "--sent-neg": "#6a2c91",     # purple
    "--sent-pos": "#1f5c99",     # blue
    "--sent-neu": "#55607a",
    # sequential scale for heatmaps (single hue, colourblind-safe); text colour per step
    "--seq-0": "#f2f5fb", "--seq-1": "#d9e3f2", "--seq-2": "#b6c9e6",
    "--seq-3": "#86a6d4", "--seq-4": "#45699f", "--seq-5": "#264e8c",
}
SEQ_STEPS = ["--seq-0", "--seq-1", "--seq-2", "--seq-3", "--seq-4", "--seq-5"]
SEQ_TEXT = ["--ink", "--ink", "--ink", "--ink", "--on-navy", "--on-navy"]

# (text variable, background variable, minimum ratio) — every pairing the CSS uses
CONTRAST_PAIRS = [
    ("--ink", "--bg", 4.5), ("--ink-2", "--bg", 4.5), ("--muted", "--bg", 4.5),
    ("--muted", "--panel", 4.5), ("--muted", "--panel-2", 4.5), ("--ink-2", "--panel", 4.5),
    ("--navy", "--bg", 4.5), ("--navy", "--panel", 4.5), ("--navy", "--panel-2", 4.5),
    ("--on-navy", "--navy", 4.5), ("--on-navy", "--navy-2", 4.5), ("--on-navy", "--bar", 4.5),
    ("--st-under", "--bg", 4.5), ("--st-under", "--st-under-bg", 4.5),
    ("--st-target", "--bg", 4.5), ("--st-target", "--st-target-bg", 4.5),
    ("--st-over", "--bg", 4.5), ("--st-over", "--st-over-bg", 4.5),
    ("--st-na", "--bg", 4.5), ("--st-na", "--st-na-bg", 4.5),
    ("--sent-neg", "--bg", 4.5), ("--sent-pos", "--bg", 4.5), ("--sent-neu", "--bg", 4.5),
    ("--ink", "--bar-track", 4.5),
] + [(t, s, 4.5) for s, t in zip(SEQ_STEPS, SEQ_TEXT)]

STATUS_CLASS = {"Under Supported": "st-under", "Target": "st-target",
                "Over Supported": "st-over", "Agreement-only": "st-na"}
STATUS_ICON = {"Under Supported": "▼", "Target": "●", "Over Supported": "▲", "Agreement-only": "◇"}

# minimum font sizes (px)
MIN_NOTE_PX_SCREEN = 11
MIN_PRINT_TABLE_PX = 9


def css_vars() -> str:
    return ":root{" + ";".join(f"{k}:{v}" for k, v in COLORS.items()) + "}"


# ---------------------------------------------------------------------------
# WCAG contrast
# ---------------------------------------------------------------------------
def _lin(c):
    c = c / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast(fg: str, bg: str) -> float:
    a, b = luminance(fg), luminance(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def parse_css_vars(css: str) -> dict:
    """Read --name:#hex pairs from a CSS string (used by the contrast test)."""
    return dict(re.findall(r"(--[\w-]+)\s*:\s*(#[0-9a-fA-F]{6})", css))


# ---------------------------------------------------------------------------
# Base stylesheet shared by both cards
# ---------------------------------------------------------------------------
BASE_CSS = """
 body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);
   max-width:880px;margin:24px auto;padding:0 22px;line-height:1.5;background:var(--bg)}
 h1{font-size:24px;margin:0 0 2px;color:var(--navy)} .sub{color:var(--muted);font-size:13px;margin-bottom:12px}
 h2{font-size:16px;color:var(--navy);margin:22px 0 6px;border-bottom:2px solid var(--rule);padding-bottom:4px;
   text-transform:uppercase;letter-spacing:.03em}
 h3{font-size:13.5px;color:var(--navy);margin:14px 0 4px}
 p,li{font-size:13px} .note{font-size:12px;color:var(--muted);margin:4px 0}
 table{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:6px}
 th,td{text-align:left;padding:4px 7px;border-bottom:1px solid var(--rule);vertical-align:top}
 th{color:var(--ink-2);font-weight:600;background:var(--panel)}
 td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
 .st{display:inline-block;font-weight:700;border-radius:4px;padding:0 5px;white-space:nowrap}
 .st-under{color:var(--st-under);background:var(--st-under-bg)}
 .st-target{color:var(--st-target);background:var(--st-target-bg)}
 .st-over{color:var(--st-over);background:var(--st-over-bg)}
 .st-na{color:var(--st-na);background:var(--st-na-bg)}
 .sev{font-weight:400;font-size:11.5px}
 .stand{font-weight:700;color:var(--navy)}
 .sent-neg{color:var(--sent-neg);font-weight:600} .sent-pos{color:var(--sent-pos);font-weight:600}
 .sent-neu{color:var(--sent-neu);font-weight:600}
 .badge{display:inline-flex;align-items:center;justify-content:center;min-width:58px;height:50px;padding:0 8px;
   border-radius:9px;background:var(--navy);color:var(--on-navy);font-size:24px;font-weight:800}
 .bars{margin:6px 0 2px}
 .bar-row{display:grid;grid-template-columns:minmax(180px,42%) 1fr;gap:8px;align-items:center;margin:4px 0}
 .bar-lab{font-size:12px;color:var(--ink)}
 .bar-track{position:relative;background:var(--bar-track);border-radius:3px;height:18px}
 .bar-fill{position:absolute;left:0;top:0;bottom:0;background:var(--bar);border-radius:3px}
 .bar-val{position:absolute;left:6px;top:0;line-height:18px;font-size:11.5px;font-weight:700;color:var(--ink)}
 .bar-axis{display:grid;grid-template-columns:minmax(180px,42%) 1fr;gap:8px;font-size:11px;color:var(--muted)}
 .bar-axis span{display:flex;justify-content:space-between}
 .kv{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12.5px} .kv b{color:var(--navy)}
 blockquote{margin:7px 0;padding:8px 12px;background:var(--panel);border-left:3px solid var(--navy-2);
   font-size:13px;font-style:italic}
 blockquote cite{display:block;font-style:normal;color:var(--muted);font-size:11.5px;margin-top:4px}
 .foot{color:var(--muted);font-size:11.5px;margin-top:22px;border-top:1px solid var(--rule);padding-top:8px}
 .print-only{display:none}
 a{color:var(--navy-2)}
 @page{size:Letter;margin:11mm}
 @media print{
   body{max-width:100%;margin:0;padding:0;font-size:12px}
   .screen-only{display:none !important} .print-only{display:block}
   h1,h2,h3{break-after:avoid;page-break-after:avoid}
   tr,blockquote,.bar-row,.keep{break-inside:avoid;page-break-inside:avoid}
   thead{display:table-header-group}
   table{font-size:10px} .note,.foot{font-size:10px}
   a{color:var(--ink);text-decoration:none}
 }
"""


def stylesheet(extra: str = "") -> str:
    return css_vars() + BASE_CSS + extra
