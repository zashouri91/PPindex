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

# Version of the keyword rules. Bump when patterns change: the optional model
# tagger (llm_tagger.py) caches by it, and the gold-set report records it.
TAXONOMY_VERSION = "2026-11-rules-v2"

# ---------------------------------------------------------------------------
# Shared regex fragments for the sense-restricted rules (Nov 2026 review)
# ---------------------------------------------------------------------------
# "desk" that is a workspace — never a front / help / service / security / reception /
# tech / concierge / information desk
_DESK = (r"(?<!front )(?<!help )(?<!service )(?<!security )(?<!reception )(?<!tech )"
         r"(?<!concierge )(?<!information )(?<!welcome )(?<!it )\bdesks?\b")
_WS = r"(?:" + _DESK + r"|\bseats?\b|workstations?|\bcubes?\b|cubicles?|\bwork ?spaces?)"
_SEP = r"[^\w.!?;]+"                    # word gap that never crosses a sentence end
_W = r"(?:" + _SEP + r"\w+){0,4}?" + _SEP   # up to four words in between, same sentence

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
    # Collaboration SPACE needs a space sense; "nobody I collaborate with is here" is a
    # presence issue and routes to In-Person / Team Presence below.
    "Collaboration Space": (
        "Collaboration & Meetings",
        [r"\bcollaborat\w*" + _W + r"(?:spaces?|areas?|rooms?|zones?|tables?|seating|settings?|pods?|hubs?|neighbou?rhoods?)\b",
         r"\bcollaborat\w* (?:spaces?|areas?|rooms?|zones?|tables?|seating|settings?|pods?|hubs?)\b",
         r"(?:spaces?|areas?|rooms?|places?|tables?|seating)\W+(?:\w+\W+){0,3}?(?:to|for|where (?:we|teams?|people) can) collaborat",
         r"team (?:spaces?|areas?|rooms?|tables?|neighbou?rhoods?)", r"group work (?:spaces?|areas?|rooms?)",
         r"open collaboration", r"brainstorm\w*(?: (?:rooms?|spaces?|areas?))?", r"white ?boards?"],
    ),
    "In-Person / Team Presence": (
        "Collaboration & Meetings",
        # "in person" only in a working-together sense (not in-person service, events, or
        # the noise of other people's in-person discussions)
        [r"\bin[\s-]?person (?:collaborat\w*|meetings?|interactions?|connections?|engagement|team ?work|"
         r"work(?:ing)? (?:with|together)|relationships?|presence|time with|contact)",
         r"(?:collaborat\w*|meet|meeting|work|working|connect\w*|interact\w*|engage|see (?:them|my team|colleagues|co-?workers|each other|anyone))"
         r"(?: with (?:\w+ ){0,2}?\w+)? in[\s-]?person\b",
         r"face[\s-]?to[\s-]?face", r"see my team", r"team is (?:not )?here",
         r"nobody (?:is )?(?:here|in)", r"empty office", r"team not in", r"coworkers? (?:aren'?t|not) (?:here|in)",
         # collaboration-with-people complaints (routed here from Collaboration Space)
         r"(?:no ?one|nobody|none of (?:my|the) \w+|not with (?:my|the) team|no team ?mates?)\W+(?:\w+\W+){0,6}?collaborat",
         r"collaborat\w*\W+(?:\w+\W+){0,6}?(?:no ?one|nobody|(?:is|are)n'?t (?:here|in)|not (?:here|in the office|on ?site)|"
         r"remote(?:ly)?|virtual(?:ly)?|other (?:locations?|states?|sites?|offices?|cities)|different (?:locations?|states?|sites?|offices?|cities))",
         r"(?:people|team|colleagues|co-?workers|those|anyone) (?:i|we) (?:work|collaborate) with (?:are|is) (?:not|never|rarely|in|located|remote|based)"],
    ),

    # --- Physical Environment -----------------------------------------------
    # Needs an availability / booking / assignment sense — a bare mention of a desk,
    # seat, cube or workstation is NOT enough (that tagged 1,459 comments, ~8% on topic).
    "Desk Availability & Reservations": (
        "Physical Environment",
        [r"(?:reserv\w*|book\w*|schedul\w*)" + _W + _WS, _WS + _W + r"(?:reserv\w*|book\w*)",
         r"\b(?:desk|seat|workstation|space|spot|cube)[ -](?:reservation|booking)s?",
         r"(?:reservation|booking) (?:system|tool|app|window|process|policy|software)",
         r"hotel(?:l?ing|led)\b", r"hot[\s-]?desk",
         r"(?:assigned|permanent|dedicated|designated|unassigned|own|personal|reserved) " + _WS,
         r"(?:find|finding|found|get|getting|secure|securing|grab|grabbing)" + _W + r"(?:a|an|any|open|available|free) (?:" + _WS + r"|spot\b)",
         r"(?:find|finding) (?:a |an )?(?:" + _WS + r"|spot\b)",
         r"(?:not enough|no|lack of|shortage of|limited|few) (?:available |open |free )?" + _WS + r"(?: available)?",
         r"more (?:available|open|free) " + _WS,
         _WS + r" (?:are|were|is|was|get|gets) (?:all |always |often |usually |completely )?(?:taken|full|booked|gone|unavailable|occupied)",
         r"(?:someone|somebody|people|others) (?:is |are |was |were )?(?:in|at|sitting (?:in|at)|using|taking) (?:my|our|a|the) (?:reserved |booked )?" + _WS,
         r"no[- ]shows?", r"first[- ]come[ ,-]+first[- ]serve", r"same " + _WS + r" every",
         # reserving / booking in general is the desk-booking sense here — except rooms and parking
         r"(?<!room )(?<!rooms )(?<!parking )(?<!conference )(?<!meeting )\breserv(?:e|ed|es|ing|ations?)\b"
         r"(?!\W+(?:\w+\W+){0,2}?(?:rooms?|conference|meeting|huddle|parking|garage|lunch|table|treadmills?|bikes?|massage|lockers?|gym))",
         r"(?<!room )(?<!rooms )(?<!parking )(?<!conference )(?<!meeting )\bbook(?:ing|ings|ed)\b"
         r"(?!\W+(?:\w+\W+){0,2}?(?:rooms?|conference|meeting|huddle|parking|garage|travel|flights?))",
         r"\bbook (?:a|my|an|the) (?:" + _WS + r"|spot|space)\b"],
    ),
    "Furniture & Ergonomics": (
        "Physical Environment",
        [r"chair", r"ergonomic", r"standing desk", r"sit[\s-]?stand", r"monitor arm",
         r"furniture", r"back pain", r"footrest", r"adjustable",
         # "uncomfortable" alone is mostly temperature / noise / social discomfort
         r"uncomfortable (?:chairs?|seats?|seating|desks?|furniture|stools?|couch\w*|sofas?|benches?)",
         r"(?:chairs?|seats?|seating|furniture|stools?) (?:are|is|were|was) (?:very |so |really |extremely )?uncomfortable"],
    ),
    "Temperature / Thermal": (
        "Physical Environment",
        [r"\bcold\b(?! (?:water|food|drinks?|brew|beverages?|calls?|sandwich\w*|items?|options?|storage|case|snacks?|foods|lunch\w*|meals?|salads?))",
         r"\bhot\b(?![- ](?:food|chocolate|water|desk\w*|meals?|lunch\w*|breakfast|drinks?|bar|items?|options?|coffee|tea|spots?|topics?|foods|entrees?|soups?|beverages?|dogs?|plates?|sauce))",
         r"temperature", r"thermal", r"freezing(?! up)", r"chilly", r"frigid", r"sweat\w*", r"stuffy",
         r"\bwarm\b(?! (?:welcome|food|meals?|smiles?|greeting|and (?:welcoming|friendly|inviting)|lunch\w*|options?|foods|drinks?|beverages?|atmosphere|feel|environment))",
         r"air condition", r"\bhvac\b", r"\ba/?c\b", r"thermostat",
         r"\bheat(?:ing|er|ers)?\b(?! (?:up|my|our|your|food|lunch\w*|meals?|things))", r"too (?:cold|hot|warm)"],
    ),
    "Lighting": (
        "Physical Environment",
        [r"\blight(?:ing|s|ed)?\b", r"\bdark\b", r"too bright", r"natural light", r"\bwindow",
         r"fluorescent", r"dim\b", r"glare"],
    ),
    "Space Availability & Crowding": (
        "Physical Environment",
        [r"crowded", r"(?:at|over|full|max|maximum|reach\w*) capacity", r"capacity (?:issues?|limits?|problems?)",
         r"packed", r"too many people", r"not enough (?:space|room|desks)",
         # "full" only in a crowding sense — not full-time / full day / full-service
         r"(?<!lot )(?<!garage )(?<!parking )(?:is|are|was|were|gets?|getting|got|been|always|often|usually|completely|too|so) full\b(?![- ](?:time|day|days|week|service|access|team|benefits?))",
         r"overcrowd", r"room to work", r"more space", r"(?<!parking )\bspaces? (?:for|to)\b"],
    ),
    "Wellbeing / Wellness Spaces": (
        "Physical Environment",
        [r"wellbeing", r"well[\s-]?being", r"wellness", r"mother'?s room", r"lactation",
         r"nursing room", r"prayer", r"meditation", r"quiet room", r"\bgym\b", r"mental health",
         r"\bnap (?:rooms?|pods?|areas?|spaces?)\b", r"\bnap ?pods?\b", r"(?:place|room|space|somewhere)s? to (?:take a )?nap\b",
         r"decompress"],
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
         r"sanitation", r"messy", r"\bdust(?:y|ing|ed)?\b", r"\bsmell", r"upkeep", r"hygien"],
    ),
    "Reception & Entry": (
        "Amenities & Services",
        # "entry" = building entry (not data entry / entry level / entry to a room); "check in"
        # only at reception (desk-reservation check-ins and "checking a box" are not reception)
        [r"reception", r"(?<!data )\bentry\b(?! (?:level|to (?:a|the|my) (?:room|space|desk))\b)(?! and departure)",
         r"\bentr(?:ance|ances|yway)\b", r"\bbadge",
         r"front desk", r"(?:visitor|guest|front desk|lobby|security) check[\s-]?ins?\b",
         r"\bcheck[\s-]?in (?:at|with) (?:the )?(?:front desk|reception|security|lobby|guard)", r"\blobby\b",
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
         r"logon", r"(?<!person )(?<!meaningful )(?<!personal )(?<!social )(?<!human )(?<!real )(?<!team )\bconnect(?!\w*\s+capsules?)(?!\w*\s+with\b)(?!ions? and collaborat)", r"\bpower\b", r"outlet", r"cabling", r"\bcable",
         r"\baccess\b.*(?:site|system|network)"],
    ),
    "Hardware (Monitors / Peripherals)": (
        "Technology",
        [r"monitor", r"keyboard", r"\bmouse\b", r"docking", r"\bdock\b", r"peripheral",
         r"adapter", r"hardware", r"equipment", r"webcam", r"headset", r"dual screen"],
    ),
    "AV / Video Conferencing": (
        "Technology",
        # an audio / video / room-technology sense is required — bare "Teams" is mostly the
        # app or "our teams" (272 of 353 comments, ~10% on topic)
        [r"video ?conferenc",
         r"\bvideo (?:quality|issues?|problems?|feeds?|lag\w*|freez\w*|technology|tech|equipment|screens?|displays?|set ?ups?|connections?|bar)\b",
         r"\baudio ?(?:/|and|&) ?video\b", r"\bzoom\b", r"camera", r"webcam", r"microphone", r"\bmics?\b",
         r"\baudio\b", r"screen ?shar", r"projector",   # "on speaker(phone)" is a noise complaint, not AV
         r"room (?:tech\w*|equipment|displays?|screens?|tvs?|av\b)", r"conference (?:tech\w*|equipment|phones?)",
         r"teams (?:rooms?|devices?|equipment|panels?|displays?|consoles?|touch ?panels?)",
         r"(?:tvs?|displays?|screens?|monitors?) (?:in|for) (?:the |our |a )?(?:conference|meeting|huddle|collaboration) rooms?",
         r"\bhdmi\b", r"\bclick ?share\b", r"\ba/?v\b(?! ?\w*\s+(?:team|department))", r"\bav\b",
         r"(?:can'?t|cannot|hard to|unable to) (?:hear|see) (?:the |other |remote )?(?:people|participants|attendees|them|others|anyone)"
         r"(?: \w+){0,2} (?:on|in|during) (?:the |a |my |our )?(?:calls?|meetings?|teams|zoom|room)"],
    ),
    "Software / Tools": (
        "Technology",
        [r"software", r"application", r"\bapp\b", r"sharepoint",
         r"(?<!hvac )(?<!\bac )(?<!a/c )(?<!heating )(?<!cooling )(?<!air )(?<!parking )(?<!building )(?<!ventilation )(?<!sprinkler )(?<!pa )(?<!p\.a\. )(?<!sound )(?<!immune )(?<!mounting )(?<!numbering )(?<!number )\bsystems?\b",
         r"reservation tool",
         r"booking (?:app|tool|system)", r"\btool", r"outlook", r"onedrive"],
    ),

    # --- Workplace Operations ------------------------------------------------
    "Building Operations": (
        "Workplace Operations",
        [r"elevator", r"escalator", r"building (?:maintenance|operations?|management|systems?|services?)",
         r"out of order", r"maintenance", r"repair", r"\bstairs\b",
         r"door(?:s)? (?:not|won'?t)",
         # "broken" only for building fixtures (broken keyboards / monitors are Hardware)
         r"(?:broken|break(?:s|ing)? down)" + _W + r"(?:elevators?|escalators?|doors?|toilets?|sinks?|faucets?|stalls?|locks?|blinds|shades|"
         r"windows?|fountains?|hand ?dryers?|lights?|lighting|fixtures?|a/?c|hvac|turnstiles?|gates?|readers?|plumbing|ice machines?)",
         r"(?:elevators?|escalators?|doors?|toilets?|sinks?|faucets?|stalls?|locks?|blinds|shades|windows?|fountains?|hand ?dryers?|lights?|"
         r"fixtures?|a/?c|hvac|turnstiles?|gates?|plumbing|ice machines?)" + _W + r"(?:broken|break(?:s)? down|not working|out of service)",
         # "facility" alone usually just means the building
         r"facilit(?:y|ies) (?:management|maintenance|team|staff|request|ticket|issues?|work orders?|operations|crew)"],
    ),
    "Parking & Commute": (
        "Workplace Operations",
        [r"parking", r"garage", r"commute", r"\btraffic\b", r"\btransit\b", r"\bdrive\b",
         r"gas money", r"mileage", r"\btoll", r"public transport"],
    ),
    "Navigation / Wayfinding": (
        "Workplace Operations",
        [r"navigat", r"wayfinding", r"finding (?:a )?(?:space|room|desk)", r"directional",
         # "sign" only in a wayfinding sense — not sign in/up/on, and not etiquette signs
         r"signage", r"(?:direction\w*|wayfinding|clear(?:er)?|better|enough|no|section|room|floor|zone|maps? (?:or|and)|missing) signs?\b(?! (?:in|on|up|off|into|out)\b)",
         r"\bsigns? (?:to direct|directing|pointing|showing (?:where|the way)|for (?:each|the) (?:sections?|rooms?|floors?|zones?|areas?))",
         r"(?:get|got|gets|getting|feel|felt|easy to get|easily) lost\b", r"\blayout\b", r"hard to find"],
    ),
    "Communication & Updates": (
        "Workplace Operations",
        [r"communicat",
         # "update(s)" only in a communication sense (most hits were software / IT updates)
         r"\bupdates? (?:on|about|regarding)\b(?! (?:my|the|our) (?:laptop|computer|software|machine|pc|applications?))",
         r"\b(?:any|no|regular|timely|frequent|status|email|e-mail|monthly|weekly|communication|project|construction) updates?\b"
         r"(?! (?:to|on) (?:my|the|our) (?:laptop|computer|software|machine|pc|applications?))",
         r"\bkeep (?:us|me|associates|everyone|staff|people) (?:\w+ )?(?:updated|posted)\b", r"notif",
         r"inform(?:ed|s|ing)? (?:us|me|staff|associates|everyone|the team|about|of|on)\b",
         r"keep .{0,20}informed", r"\baware", r"announcement",
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
        [r"\brto\b", r"return to office", r"work from home", r"\bwfh\b",
         # "remote" / "hybrid" only in a work-arrangement sense — not "remote colleagues",
         # "hybrid huddle" / "hybrid office" room names or "hybrid colleagues"
         r"\b(?:i|we|to|prefer to|could|can|would rather|allowed to|able to|option to) (?:work|working|be|go|stay) (?:\w+ )?remote(?:ly)?\b",
         r"\b(?:full(?:y|-time| time)?|100%|permanent(?:ly)?|completely|totally|back to|return to|go) remote(?:ly)?\b"
         r"(?! (?:colleagues?|team ?mates?|teams?|members?|co-?workers?|associates?|staff|people|participants?))",
         r"\bremote (?:work|working|option|options|flexibility|days?|status|roles?|positions?|eligibility|exceptions?|arrangements?|policy)\b",
         r"\b(?:than|vs\.?|versus|compared to) (?:working )?remote(?:ly)?\b",
         r"mandate", r"in[\s-]?office (?:days?|requirements?|policy|mandate|expectations?|schedule)",
         r"days (?:in|per week)", r"come in(?:to)? the office",
         r"hybrid (?:schedule|policy|model|work\w*|arrangement|days?|week|mandate|requirements?|approach|environment|employee|role|status|expectations?)",
         r"\b(?:i am|i'?m|being|as) (?:a )?hybrid\b(?! (?:colleagues?|team|teams|members?|people|associates?|co-?workers?|huddles?|rooms?|offices?|spaces?))",
         r"more productive at home", r"forced to"],
    ),
    "Positive / Appreciation": (
        "Policy / Experience",
        [r"(?<!would be )(?<!it'd be )(?<!not )(?<!be )(?<!isn't )(?<!wasn't )\bgreat\b(?! if)",
         r"(?<!would )(?<!i'd )(?<!not )(?<!don't )(?<!do not )\blove\b(?! (?:to|if|it if|for|more|some|a|an|better|the option|the ability))",
         r"excellent", r"appreciate", r"wonderful",
         r"(?<!would be )(?<!at )(?<!do our )(?<!do my )(?<!try our )(?<!not the )\bbest\b(?! (?:to|if|practices?|way to|option would))",
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

# Generic buckets: keyword catch-alls with no specific, actionable content.
# Excluded wherever themes are ranked or rolled up for priorities (site card,
# portfolio, Action Priorities, key drivers, hotspots, action-plan payload).
# Their comments still count toward sentiment totals and the comment browser.
GENERIC_THEMES = frozenset({"Positive / Appreciation", "Overall Dissatisfaction"})

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
