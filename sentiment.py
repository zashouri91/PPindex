"""
Domain-tuned ensemble sentiment for PulsePoint workplace comments.

Why not plain VADER: VADER is a generic social-media lexicon. On facilities
feedback it misreads domain vocabulary ("the lack of privacy limits us",
"it's always freezing", "parking is a nightmare") and constructively-phrased
complaints. This module combines:

  1. VADER compound            (general polarity + grammar/negation)
  2. A facilities lexicon      (workplace-specific pos/neg terms & phrases)
  3. Negation + intensifiers   (flip / scale nearby cues)
  4. A light question-framing prior (Q13 asks what "limits" you -> leans neg)
  5. Explicit MIXED detection  (strong pos AND strong neg in the same comment)

Returns a signed score in [-1, 1] and one of:
  "Positive" | "Negative" | "Neutral / Mixed"

Pure-Python, no downloads — works fully offline.
"""
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()

# ---------------------------------------------------------------------------
# Facilities / workplace lexicon.  weight = base magnitude (>1 = stronger cue)
# ---------------------------------------------------------------------------
NEG_TERMS = {
    # comfort / environment
    "cold":1.2, "freezing":1.6, "frigid":1.5, "chilly":1.0, "hot":1.0, "stuffy":1.2,
    "uncomfortable":1.3, "loud":1.3, "noisy":1.3, "noise":1.0, "distracting":1.3,
    "cramped":1.4, "crowded":1.3, "packed":1.1, "dark":0.8, "dingy":1.2, "smell":1.1,
    "smelly":1.4, "dirty":1.5, "filthy":1.7, "unclean":1.4, "gross":1.4, "messy":1.1,
    # availability / capacity
    "lack":1.3, "lacking":1.3, "limits":1.2, "limited":1.2, "limiting":1.3,
    "insufficient":1.4, "inadequate":1.4, "unavailable":1.3, "scarce":1.3,
    "shortage":1.4, "cannot":0.9, "unable":1.2, "difficult":1.2, "hard":0.8,
    "struggle":1.4, "struggling":1.4, "impossible":1.5,
    # cost
    "expensive":1.3, "overpriced":1.6, "pricey":1.2, "costly":1.2, "unaffordable":1.5,
    # reliability / ops
    "broken":1.5, "malfunction":1.4, "unreliable":1.5, "slow":1.1, "outdated":1.3,
    "glitch":1.2, "glitchy":1.3, "stolen":1.3, "missing":1.2, "wait":0.7,
    "delay":1.1, "delayed":1.1, "outage":1.4, "disconnect":1.2, "frozen":1.1,
    # affect
    "frustrating":1.6, "frustrated":1.6, "frustration":1.5, "nightmare":1.8,
    "terrible":1.8, "awful":1.8, "horrible":1.8, "poor":1.3, "disappointing":1.5,
    "disappointed":1.5, "annoying":1.4, "annoyed":1.3, "hate":1.7, "dread":1.5,
    "dreaded":1.5, "miserable":1.7, "unhappy":1.4, "concern":0.8, "concerned":0.9,
    "complaint":1.0, "complaints":1.0, "issue":0.8, "issues":0.9, "problem":1.0,
    "problems":1.1, "problematic":1.3, "waste":1.4, "pointless":1.5, "useless":1.5,
    "worse":1.3, "worst":1.6, "bad":1.2, "unproductive":1.4, "inconsistent":1.1,
    "hinder":1.4, "hinders":1.4, "hindered":1.4, "hindering":1.4, "irrelevant":1.3,
    "isolated":1.3, "isolating":1.4, "faulty":1.4, "empty":1.0, "nobody":1.0,
    "overcrowded":1.4, "understaffed":1.4, "clunky":1.2, "cumbersome":1.3,
}
POS_TERMS = {
    "great":1.4, "love":1.6, "loved":1.5, "excellent":1.7, "amazing":1.7,
    "awesome":1.6, "wonderful":1.6, "fantastic":1.7, "perfect":1.6, "outstanding":1.6,
    "nice":1.0, "good":1.0, "convenient":1.3, "comfortable":1.3, "spacious":1.4,
    "clean":1.2, "quiet":1.1, "bright":0.9, "modern":1.1, "welcoming":1.3,
    "pleasant":1.3, "friendly":1.2, "helpful":1.3, "efficient":1.2, "reliable":1.3,
    "easy":1.1, "seamless":1.4, "appreciate":1.4, "appreciated":1.4, "enjoy":1.4,
    "enjoyed":1.4, "enjoyable":1.4, "happy":1.3, "satisfied":1.3, "impressed":1.4,
    "best":1.4, "beautiful":1.4, "productive":1.2, "accommodating":1.3,
    "flexible":1.0, "positive":1.1, "grateful":1.4, "thankful":1.3, "smooth":1.2,
    "superb":1.6,
    # NOTE: "better"/"more"/"improved" are deliberately NOT positive here — in this
    # survey "better X" / "more X" almost always signals a current shortfall (a
    # request), handled as an unmet-need negative below.
}
# multiword phrases -> (weight, polarity)  polarity: -1 neg, +1 pos
# Phrase matches are stripped from the text before the single-word pass, so a
# hypothetical positive inside a request ("would be helpful") is counted as an
# unmet-need negative and does NOT also fire the positive word "helpful".
PHRASES = {
    # concrete problems
    r"out of order":(1.5,-1), r"lack of":(1.4,-1), r"not enough":(1.5,-1),
    r"no privacy":(1.7,-1), r"too cold":(1.5,-1), r"too hot":(1.3,-1),
    r"too loud":(1.5,-1), r"too expensive":(1.5,-1), r"too crowded":(1.4,-1),
    r"hard to (?:find|book|reserve|hear|focus|concentrate|get)":(1.5,-1),
    r"difficult to (?:find|book|reserve|hear|focus|concentrate|get)":(1.5,-1),
    r"never works?":(1.6,-1), r"(?:doesn'?t|does not|don'?t|do not|won'?t|will not) work":(1.5,-1),
    r"not work(?:ing)?":(1.4,-1), r"waste of (?:time|money|space)":(1.6,-1),
    r"more productive (?:at|from) home":(1.4,-1), r"are (?:all )?taken":(1.2,-1),
    r"(?:is|are|been) broken":(1.5,-1), r"under maintenance":(1.3,-1),
    # request / wish framing -> unmet need (negative). these also strip the
    # hypothetical positive word so it isn't double-counted.
    r"would be (?:very |really |much )?(?:nice|helpful|good|great|useful|beneficial|better|convenient)":(1.1,-1),
    r"it would (?:be )?(?:nice|help|helpful|good|great)":(1.1,-1),
    r"would (?:like|prefer|love) (?:to|it|more|a|the|if|having|access)":(1.0,-1),
    r"i wish":(1.2,-1), r"needs? (?:to|more|better|improvement)":(1.2,-1),
    r"should (?:be able|have|allow|provide|include|not|consider|offer)":(1.1,-1),
    r"could be better":(1.2,-1), r"room for improvement":(1.2,-1),
    r"more (?:space|room|privacy|quiet|option|options|monitor|monitors|desk|desks|seating|parking|variety|choices?|rooms?)":(1.1,-1),
    r"better (?:option|options|food|parking|monitor|monitors|chair|chairs|reservation|discretion|lighting|noise|space|spaces?|policy|signage|equipment|service|coffee)":(1.1,-1),
    # positives
    r"work(?:s)? (?:great|well|perfectly)":(1.4,1), r"easy to (?:use|find|book|navigate|access)":(1.3,1),
    r"love (?:the|my|coming|working|going|being)":(1.6,1), r"really (?:nice|enjoy|good|great|helpful)":(1.4,1),
    r"(?:no|without any) (?:complaints?|issues?|problems?)":(1.4,1), r"well[- ]designed":(1.4,1),
    r"state of the art":(1.4,1), r"very (?:nice|helpful|clean|comfortable|convenient|happy|pleased)":(1.4,1),
    r"great (?:job|place|space|experience|office|team|people|staff)":(1.5,1),
    # weak-negative office-value / isolation signals
    r"no one":(1.0,-1), r"very little":(1.0,-1), r"no collaboration":(1.3,-1),
    r"(?:don'?t|do not) understand":(1.0,-1),
}

# strong standalone positives that should protect a comment from request-dampening
STRONG_POS = {"love","loved","excellent","amazing","awesome","wonderful","fantastic",
              "perfect","outstanding","superb","great","best"}
CLAUSE_SPLIT = re.compile(r"[,.;:!?]|\band\b|\bbut\b|\bhowever\b|\balthough\b|"
                          r"\bthough\b|\bthat\b|\bwhich\b|\bwhere\b|\bso\b")

# request/wish lead-ins: a positive cue in this context is a hypothetical wish
# (an unmet need), so it should read negative, not positive.
WISH_LEADS = {"would","wish","hope","need","needs","needed","should","prefer",
              "prefers","request","requests","suggest","suggestion","recommend",
              "want","wants","wanted","lacking","more","additional","better"}

NEGATORS = {"not","no","never","without","cannot","can't","cant","don't","dont",
            "doesn't","doesnt","isn't","isnt","aren't","arent","wasn't","wasnt",
            "won't","wont","hardly","rarely","neither","nor","lack","lacking"}
INTENSIFIERS = {"very":1.5,"extremely":1.7,"really":1.35,"so":1.25,"too":1.3,
                "incredibly":1.6,"absolutely":1.5,"super":1.4,"quite":1.2,
                "highly":1.4,"totally":1.4,"completely":1.4,"always":1.2}
DIMINISHERS = {"slightly":0.6,"somewhat":0.7,"a little":0.6,"kind of":0.7,
               "kinda":0.7,"sort of":0.7,"barely":0.6,"a bit":0.65,"mostly":0.85}

# question-framing prior (weak nudge, not an override)
QUESTION_PRIOR = {
    "Physical Limitations":       -0.12,  # "what most LIMITS your ability"
    "Work Types Not Supported":   -0.06,  # asks for gaps
    "Services (improve/hinder)":   0.0,
    "Technology (improve/hinder)": 0.0,
    "Overall Experience":          0.0,
}

_word_re = re.compile(r"[a-z']+")

def _lexicon_signal(text):
    """Return (pos_weight, neg_weight) from domain lexicon w/ negation, intensifiers,
    and request/wish-context handling."""
    t = " " + text.lower() + " "
    pos_w = neg_w = 0.0
    # phrase pass — accumulate, then blank the span so the token pass won't
    # double-count words inside a matched phrase (e.g. "helpful" in "would be helpful")
    for pat,(w,pol) in PHRASES.items():
        def _repl(m):
            nonlocal pos_w, neg_w
            if pol > 0: pos_w += w
            else:       neg_w += w
            return " " * (m.end()-m.start())
        t = re.sub(pat, _repl, t)
    has_strong_pos = any(re.search(r"\b%s\b" % w, t) for w in STRONG_POS)
    # token pass, CLAUSE BY CLAUSE so negation never crosses a clause boundary
    # (e.g. "not broken and missing" must not flip "missing")
    for clause in CLAUSE_SPLIT.split(t):
        toks = _word_re.findall(clause)
        for i, tok in enumerate(toks):
            base = None; polarity = 0
            if tok in NEG_TERMS: base, polarity = NEG_TERMS[tok], -1
            elif tok in POS_TERMS: base, polarity = POS_TERMS[tok], 1
            if base is None:
                continue
            mult = 1.0; neg_flip = False; wish_ctx = False
            for j in range(max(0, i-3), i):
                pj = toks[j]
                if pj in NEGATORS: neg_flip = not neg_flip
                if pj in INTENSIFIERS: mult *= INTENSIFIERS[pj]
                if pj in DIMINISHERS: mult *= DIMINISHERS[pj]
                if pj in WISH_LEADS: wish_ctx = True
            eff = base * mult
            if neg_flip: polarity *= -1
            if polarity > 0 and wish_ctx:      # positive inside a wish = unmet need
                polarity = -1; eff *= 0.8
            if polarity > 0: pos_w += eff
            else:            neg_w += eff
    # standalone request cues add a mild unmet-need signal
    for cue in ("more", "better", "additional"):
        neg_w += 0.5 * t.count(" %s " % cue)
    # request-context dampening: suggestions/wishes are about shortfalls, so
    # discount incidental positives — unless the comment is strongly positive
    request_ctx = any(m in t for m in (" need", " should", " wish", " would like",
                      " suggestion", " please ", " prefer", " lacking", " allow ",
                      " more ", " better ", " additional ")) or neg_w > pos_w
    if request_ctx and not has_strong_pos:
        pos_w *= 0.55
    return pos_w, neg_w


def score_sentiment(text, question_domain=None):
    """Return (score in [-1,1], label). label in Positive/Negative/Neutral / Mixed."""
    if text is None or not str(text).strip():
        return 0.0, "Neutral / Mixed"
    txt = str(text)
    vader = _vader.polarity_scores(txt)["compound"]          # [-1,1]
    pos_w, neg_w = _lexicon_signal(txt)
    lex = (pos_w - neg_w) / (pos_w + neg_w + 1.5)             # smoothed, [-1,1]

    prior = QUESTION_PRIOR.get(question_domain, 0.0)
    combined = 0.45 * vader + 0.47 * lex + prior
    combined = max(-1.0, min(1.0, combined))

    # genuine MIXED: both sides strong and neither dominates
    strong_both = pos_w >= 1.3 and neg_w >= 1.3
    balanced = min(pos_w, neg_w) / (max(pos_w, neg_w) + 1e-9) >= 0.55 if max(pos_w,neg_w)>0 else False
    if strong_both and balanced:
        return round(combined, 4), "Neutral / Mixed"

    if combined >= 0.15:
        return round(combined, 4), "Positive"
    if combined <= -0.10:
        return round(combined, 4), "Negative"
    return round(combined, 4), "Neutral / Mixed"
