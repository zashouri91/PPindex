"""
Theme taxonomy for the PulsePoint open-ended response analyzer.

Each theme maps to:
  - a parent CATEGORY (aligned to the survey's rating domains), and
  - a list of case-insensitive keyword / regex patterns.

A comment is MULTI-LABEL: it receives every theme whose pattern matches.
This is how multi-topic comments get "split" (counted under each theme they
touch) and how granular details "roll up" into a parent theme
(e.g. "the 3rd-floor east bathroom" -> Cleanliness & Upkeep).

Keep this file the single source of truth: both build_data.py and the
Streamlit app import from here, so re-tagging is always consistent.
"""

import re

# ---------------------------------------------------------------------------
# Categories (parent domains) — aligned to the survey's rating blocks
# ---------------------------------------------------------------------------
CATEGORIES = [
    "Focused / Private Work",
    "Collaboration & Meetings",
    "Physical Environment",
    "Amenities & Services",
    "Technology",
    "Workplace Operations",
    "Policy / Experience",
]

# ---------------------------------------------------------------------------
# Theme -> (category, [patterns])
# Patterns are regex fragments, matched case-insensitively with word-ish
# boundaries. Order matters only for display; matching is independent.
# ---------------------------------------------------------------------------
THEMES = {
    # --- Focused / Private Work ---------------------------------------------
    "Privacy & Confidentiality": (
        "Focused / Private Work",
        [r"privacy", r"private", r"confidential", r"discretion", r"discreet",
         r"overhear", r"hipaa", r"\bpii\b", r"sensitive (?:info|matter|conversation|topic|nature)",
         r"phone booth", r"booth", r"closed door", r"closed-door"],
    ),
    "Noise & Distraction": (
        "Focused / Private Work",
        [r"noise", r"noisy", r"\bloud\b", r"distract", r"\bquiet\b", r"chatter",
         r"\becho", r"can'?t (?:focus|concentrate|hear)", r"too loud", r"background noise",
         r"sound", r"talking"],
    ),
    "Focus / Concentration": (
        "Focused / Private Work",
        [r"focus", r"concentrat", r"heads?[\s-]?down", r"deep work", r"uninterrupted",
         r"interruption", r"productive individual"],
    ),

    # --- Collaboration & Meetings -------------------------------------------
    "Meeting & Conference Rooms": (
        "Collaboration & Meetings",
        [r"meeting room", r"conference room", r"huddle", r"breakout", r"book(?:ing)? a room",
         r"reserve a room", r"war room", r"boardroom", r"\bmeeting space"],
    ),
    "Collaboration Space": (
        "Collaboration & Meetings",
        [r"collaborat", r"team space", r"work(?:ing)? together", r"group work",
         r"open collaboration", r"brainstorm", r"spontaneous"],
    ),
    "In-Person / Team Presence": (
        "Collaboration & Meetings",
        [r"in[\s-]?person", r"face[\s-]?to[\s-]?face", r"see my team", r"team is (?:not )?here",
         r"nobody (?:is )?(?:here|in)", r"empty office", r"team not in", r"coworkers? (?:aren'?t|not) (?:here|in)"],
    ),

    # --- Physical Environment -----------------------------------------------
    "Desk Availability & Reservations": (
        "Physical Environment",
        [r"\bdesk", r"reservation", r"reserve", r"hotel(?:ing|led)?", r"hot[\s-]?desk",
         r"\bseat", r"no desks", r"assigned (?:desk|seat)", r"find a (?:desk|seat|spot)",
         r"workstation", r"cubicle", r"\bcube\b"],
    ),
    "Furniture & Ergonomics": (
        "Physical Environment",
        [r"chair", r"ergonomic", r"standing desk", r"sit[\s-]?stand", r"monitor arm",
         r"furniture", r"uncomfortable", r"back pain", r"footrest", r"adjustable"],
    ),
    "Temperature / Thermal": (
        "Physical Environment",
        [r"\bcold\b", r"\bhot\b", r"temperature", r"thermal", r"freezing", r"chilly",
         r"\bwarm\b", r"air condition", r"\bhvac\b", r"\bheat(?:ing)?\b", r"too (?:cold|hot|warm)"],
    ),
    "Lighting": (
        "Physical Environment",
        [r"light(?:ing)?", r"\bdark\b", r"too bright", r"natural light", r"\bwindow",
         r"fluorescent", r"dim\b", r"glare"],
    ),
    "Space Availability & Crowding": (
        "Physical Environment",
        [r"crowded", r"capacity", r"packed", r"\bfull\b", r"too many people", r"not enough (?:space|room|desks)",
         r"overcrowd", r"room to work", r"more space", r"\bspace(?:s)? (?:for|to)"],
    ),
    "Wellbeing / Wellness Spaces": (
        "Physical Environment",
        [r"wellbeing", r"well[\s-]?being", r"wellness", r"mother'?s room", r"lactation",
         r"nursing room", r"prayer", r"meditation", r"quiet room", r"\bgym\b", r"mental health",
         r"nap", r"decompress"],
    ),

    # --- Amenities & Services -----------------------------------------------
    "Food & Beverage": (
        "Amenities & Services",
        [r"\bfood\b", r"cafe", r"cafeteria", r"coffee", r"snack", r"beverage", r"\blunch\b",
         r"kitchen", r"vending", r"\bmicrowave", r"\bwater\b", r"pantry", r"\bsoda\b",
         r"food truck", r"grab and go", r"grab[\s-]?n[\s-]?go"],
    ),
    "Cost / Pricing": (
        "Amenities & Services",
        [r"expensive", r"overpriced", r"\bpricing\b", r"\bprice", r"\bcost\b", r"afford",
         r"cheaper", r"\btoo much\b.*(?:food|coffee|price)"],
    ),
    "Cleanliness & Upkeep": (
        "Amenities & Services",
        [r"\bclean", r"\bdirty\b", r"bathroom", r"restroom", r"\btrash\b", r"garbage",
         r"sanitation", r"messy", r"dust", r"\bsmell", r"upkeep", r"hygien"],
    ),
    "Reception & Entry": (
        "Amenities & Services",
        [r"reception", r"\bentry\b", r"\bbadge", r"front desk", r"check[\s-]?in", r"\blobby\b",
         r"\bsecurity\b(?! camera)", r"greet"],
    ),
    "Community & Events": (
        "Amenities & Services",
        [r"community manager", r"community space", r"\bevent", r"gathering", r"social",
         r"culture", r"engagement activit"],
    ),

    # --- Technology ----------------------------------------------------------
    "Connectivity (WiFi / Login / Power)": (
        "Technology",
        [r"wi[\s-]?fi", r"wireless", r"internet", r"\bnetwork\b", r"\bvpn\b", r"log[\s-]?in",
         r"logon", r"\bconnect", r"\bpower\b", r"outlet", r"cabling", r"\bcable", r"\baccess\b.*(?:site|system|network)"],
    ),
    "Hardware (Monitors / Peripherals)": (
        "Technology",
        [r"monitor", r"keyboard", r"\bmouse\b", r"docking", r"\bdock\b", r"peripheral",
         r"adapter", r"hardware", r"equipment", r"webcam", r"headset", r"dual screen"],
    ),
    "AV / Video Conferencing": (
        "Technology",
        [r"video conferenc", r"\bteams\b", r"\bzoom\b", r"camera", r"microphone", r"\bmic\b",
         r"screen shar", r"room technology", r"\bav\b", r"projector", r"conference tech"],
    ),
    "Software / Tools": (
        "Technology",
        [r"software", r"application", r"\bapp\b", r"sharepoint", r"\bsystem", r"reservation tool",
         r"booking (?:app|tool|system)", r"\btool", r"outlook", r"onedrive"],
    ),

    # --- Workplace Operations ------------------------------------------------
    "Building Operations": (
        "Workplace Operations",
        [r"elevator", r"escalator", r"\bbuilding\b", r"out of order", r"\bbroken\b",
         r"maintenance", r"repair", r"\bstairs\b", r"door(?:s)? (?:not|won'?t)", r"facilit"],
    ),
    "Parking & Commute": (
        "Workplace Operations",
        [r"parking", r"garage", r"commute", r"\btraffic\b", r"transit", r"\bdrive\b",
         r"gas money", r"mileage", r"\btoll", r"public transport"],
    ),
    "Navigation / Wayfinding": (
        "Workplace Operations",
        [r"navigat", r"wayfinding", r"finding (?:a )?(?:space|room|desk)", r"directional",
         r"signage", r"\bsigns?\b", r"\blost\b", r"\blayout\b", r"hard to find"],
    ),
    "Communication & Updates": (
        "Workplace Operations",
        [r"communicat", r"\bupdates?\b", r"notif", r"\binform", r"\baware", r"announcement",
         r"didn'?t know", r"no idea", r"unclear"],
    ),
    "Support / Help Services": (
        "Workplace Operations",
        [r"help desk", r"\bit support", r"tech support", r"support staff", r"concierge",
         r"get help", r"resolve", r"someone to help"],
    ),

    # --- Policy / Experience -------------------------------------------------
    "RTO / Hybrid Policy": (
        "Policy / Experience",
        [r"\brto\b", r"return to office", r"work from home", r"\bwfh\b", r"\bremote\b",
         r"mandate", r"in[\s-]?office", r"days (?:in|per week)", r"hybrid", r"come in(?:to)? the office",
         r"more productive at home", r"forced to"],
    ),
    "Positive / Appreciation": (
        "Policy / Experience",
        [r"\bgreat\b", r"\blove\b", r"excellent", r"appreciate", r"wonderful", r"\bbest\b",
         r"\bhappy\b", r"\benjoy", r"fantastic", r"no complaint", r"very good", r"\bawesome"],
    ),
    "Overall Dissatisfaction": (
        "Policy / Experience",
        [r"waste of", r"pointless", r"frustrat", r"\bhate\b", r"terrible", r"\bawful\b",
         r"\bmiserable", r"\bdreaded?\b", r"unhappy", r"morale"],
    ),
}

# Pre-compile for speed
_COMPILED = {
    theme: (cat, [re.compile(p, re.IGNORECASE) for p in pats])
    for theme, (cat, pats) in THEMES.items()
}

THEME_TO_CATEGORY = {t: c for t, (c, _) in THEMES.items()}

# Junk / no-comment detection
_JUNK_EXACT = {
    "", "no", "no.", "none", "na", "n/a", "nope", "nothing", "nada", "no thanks",
    "no thank you", "no thank you!", "no, thank you", "not really", "not at all",
    "nothing else", "nothing to add", "no comment", "no comments", "all good",
    "thank you", "thanks", "thank you!", "great", "good", "ok", "okay", ".", "-",
    "n/a.", "none.", "no answer", "not sure", "idk", "no additional", "same as above",
    "see above", "see previous", "as above", "no other", "nil",
}
_JUNK_PATTERNS = [
    re.compile(r"^(see (my )?(prev(ious)?|above)( comment)?\.?)$", re.I),
    re.compile(r"^(same as (above|previous)\.?)$", re.I),
    re.compile(r"^(no+\.?|na+\.?|none+\.?|nope+\.?)$", re.I),
    # no-content answers: "nothing/none/not at this/the time|moment", optionally trailing
    re.compile(r"^(nothing|none|not|no)( (else|further|more|additional|other))?"
               r"( (that|to) (add|share|mention|note|report))?"
               r"( (at )?(this|the) (time|moment|point))?\.?$", re.I),
    re.compile(r"^(nothing|none)( (at )?(this|the) (time|moment|point))\.?$", re.I),
    re.compile(r"^(nothing (comes to mind|i can think of|else at this time))\.?$", re.I),
    re.compile(r"^(no (additional|other|further) (comment|comments|feedback|thoughts))\.?$", re.I),
]


def is_junk(text: str) -> bool:
    """True if the comment carries no analyzable content."""
    if text is None:
        return True
    t = str(text).strip().lower()
    if t in _JUNK_EXACT:
        return True
    # very short with no letters, or single non-informative token
    stripped = re.sub(r"[^a-z]", "", t)
    if len(stripped) < 3:
        return True
    for pat in _JUNK_PATTERNS:
        if pat.match(t):
            return True
    return False


def tag_themes(text: str):
    """Return (themes:list, categories:list) for a comment. Multi-label."""
    if not text or is_junk(text):
        return [], []
    themes = []
    for theme, (cat, pats) in _COMPILED.items():
        for p in pats:
            if p.search(text):
                themes.append(theme)
                break
    cats = sorted({THEME_TO_CATEGORY[t] for t in themes})
    return themes, cats
