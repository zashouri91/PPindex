"""
Hand-written replacements for stored-plan lines whose wording (rank, "highest of all
sites", "highest-volume theme", "most-negative theme") the corrected data or the
standing rule no longer supports. Used by tools/refresh_plan_evidence.py; placeholders
are filled from the corrected data at refresh time. No comment text.
"""

def _stat(k):
    return ("{" + k + "_people} people ({" + k + "_pct:.0f}% of respondents), {" + k + "_comments} comments, {"
            + k + "_neg:.0f}% negative")


def _protect(k, tail=""):
    return ("{" + k + "_pos:.0f}% positive and {" + k + "_neg:.0f}% negative across {" + k + "_comments} comments "
            "from {" + k + "_people} people" + tail)


OVERRIDES = {
    # ---- Woodland Hills
    ("ca_woodland_hills", "Friction for focused individual work is 37/100 — 64% feel under-supported; discretion 27/100 (50%)."):
        "Friction for focused individual work is 37/100 — 64% under-supported; discretion 27/100 (50% under-supported).",
    ("ca_woodland_hills", "Desk Availability & Reservations is the highest-volume theme: 174 comments, 53% negative."):
        "Desk Availability & Reservations: " + _stat("deskavaila") + " — the 5th most-raised theme by people (it was "
        "the highest-volume theme before the November 2026 re-tag, which stopped counting every mention of a desk).",
    ("ca_woodland_hills", "Woodland Hills is the lowest-ranked site overall (10 of 10); these priorities target its largest, most-cited gaps first."):
        "Woodland Hills is below the company average overall (Index {index:.0f} ± {index_margin} vs {company_index:.1f}); "
        "these priorities target its largest, most-cited gaps first.",
    ("ca_woodland_hills", "91% positive across 22 comments — security/front-desk is a genuine strength"):
        _protect("receptione", " — most comments are neutral or mixed rather than praise"),
    # ---- Miami
    ("fl_miami", "Connectivity issues undercut an otherwise top-tier site (Miami is #2 overall)."):
        "Connectivity issues undercut an otherwise well-rated site (Index {index:.0f} ± {index_margin}, in line with "
        "the company average).",
    ("fl_miami", "Miami is a top-tier site (2 of 10); these are targeted refinements, not turnaround actions."):
        "Miami is in line with the company average overall (Index {index:.0f} ± {index_margin}); these are targeted "
        "refinements, not turnaround actions.",
    ("fl_miami", "mixed: strong for many (72% positive, 18 comments) but a vocal reliability minority — protect the baseline while fixing the gaps"):
        "mixed: " + _protect("connectivi", " — a reliability minority is vocal; protect the baseline while fixing the gaps"),
    # ---- Tampa
    ("fl_tampa", "Desk Availability & Reservations: 77 comments, 48% negative (highest-volume theme)."):
        "Desk Availability & Reservations: " + _stat("deskavaila") + " — the 5th most-raised theme by people (it was "
        "the highest-volume theme before the November 2026 re-tag).",
    # ---- Atlanta
    ("ga_atlanta", "Desk Availability & Reservations: 98 comments, 45% negative — the highest-volume theme at the site not already covered by P1–P3."):
        "Desk Availability & Reservations: " + _stat("deskavaila") + " — the 2nd most-raised theme by people not "
        "already covered by P1–P3 (after Food & Beverage).",
    ("ga_atlanta", "highest of all sites (82.5/100) — the whole service experience is a model to protect"):
        "{sl:.1f} ± {sl_m}, above the company average ({sl_co:.1f}) — the whole service experience is a model to protect",
    ("ga_atlanta", "Atlanta ranks #1 overall; these are optimizations of an already-strong site, not fixes for systemic problems."):
        "Atlanta is above the company average overall (Index {index:.0f} ± {index_margin} vs {company_index:.1f}); these "
        "are optimizations of an already-strong site, not fixes for systemic problems.",
    # ---- Chicago
    ("il_chicago_willis_tower", "Work Activity is the site's weakest dimension (60.5, rank 8) despite strong Service and Physical scores."):
        "Work Activity is the site's weakest dimension ({wa:.1f} ± {wa_m}, in line with the company {wa_co:.1f}), while "
        "Service Level is above the company average ({sl:.1f} vs {sl_co:.1f}).",
    ("il_chicago_willis_tower", "81/100, rank 2 — a strength (food & beverage availability is the exception — see priority 5)"):
        "{sl:.0f} ± {sl_m}, above the company average ({sl_co:.1f}) — a strength (food & beverage availability is the "
        "exception — see priority 5)",
    # ---- Indianapolis
    ("in_indianapolis", "Digital & Tech dimension is 69.9 — rank 10 of 10."):
        "Digital & Tech dimension is {dt:.1f} ± {dt_m}, below the company average ({dt_co:.1f}).",
    ("in_indianapolis", "Indianapolis is the largest site by comment volume (1,266); the priorities target the highest-volume, highest-friction gaps."):
        "Indianapolis is the largest site by comment volume ({n_comments:,} comments from {n_commenters} people); the "
        "priorities target the highest-volume, highest-friction gaps.",
    # ---- New York
    ("ny_new_york_penn1", "70.8/100, rank 2 — a strong physical space overall (thermal comfort is the exception — see priority 4)"):
        "{pf:.1f} ± {pf_m}, above the company average ({pf_co:.1f}) — a strong physical space overall (thermal comfort "
        "is the exception — see priority 4)",
    ("ny_new_york_penn1", "New York rates well overall (3 of 10) but comment sentiment skews negative (53%); the strongest signal is the concentrated pod-capacity and Wi-Fi issues."):
        "New York is in line with the company average overall (Index {index:.0f} ± {index_margin}) but comment "
        "sentiment skews negative ({neg_pct:.0f}% of comments); the strongest signal is the concentrated pod-capacity "
        "and Wi-Fi issues.",
    # ---- Norfolk
    ("va_norfolk", "Noise & Distraction: 125 comments, 75% negative — the most-negative theme at the site."):
        "Noise & Distraction: " + _stat("noisedistr") + " — the 2nd most-negative theme at the site (after Lighting).",
    ("va_norfolk", "Norfolk is a mid-tier site (5 of 10) that's well-liked; the priorities target its few sharp negatives (noise especially)."):
        "Norfolk is in line with the company average overall (Index {index:.0f} ± {index_margin}); the priorities "
        "target its few sharp negatives (noise especially).",
    # ---- Richmond
    ("va_richmond", "On-site food/beverage availability friction is 25/100 (46% under-supported); Service Level is the site's weakest dimension (68.4, rank 10)."):
        "On-site food/beverage availability friction is 25/100 (46% under-supported); Service Level is below the "
        "company average ({sl:.1f} ± {sl_m} vs {sl_co:.1f}).",
    ("va_richmond", "Richmond is near the bottom overall (9 of 10) with the weakest Service Level; food/beverage is an unusually strong signal here vs. other sites."):
        "Richmond is below the company average overall (Index {index:.0f} ± {index_margin} vs {company_index:.1f}) and on "
        "Service Level ({sl:.1f} vs {sl_co:.1f}); food and beverage is raised by {foodbevera_pct:.0f}% of respondents "
        "here, more than at the other sites beyond chance.",
}

PRIORITY_FLAGS = {
    "in_indianapolis": [
        "Priority 2 (Desk reservations and booking window): the two-week booking window it centres on is mentioned in "
        "11 comments from 9 people; the figure it cited (375 comments) counted every mention of a desk. After the re-tag "
        "Desk Availability & Reservations is still raised by 147 people (21% of respondents), so desk booking remains "
        "material, but the booking-window change itself rests on a small number of comments — validate locally first."],
    "ca_woodland_hills": [
        "Priority 5 (Meeting-room and video-conferencing reliability): the AV / Video Conferencing comments behind it "
        "fall from 39 to 14 (14 people) once bare mentions of Microsoft Teams stop counting; the priority now rests on "
        "the Under Supported virtual-collaboration item and Meeting & Conference Rooms comments."],
    "oh_mason": [
        "Priority 5 (Meeting-room booking discipline and in-room connections): AV / Video Conferencing comments fall "
        "from 35 to 10 (10 people) after the re-tag; the in-room connections half of this priority is thinly "
        "supported by comments — the booking-discipline half (Meeting & Conference Rooms, 44 people) is not affected."],
    "fl_miami": [
        "Priority 3 (Network connectivity reliability): connectivity comments are 35% negative (17 comments from 12 "
        "people) and the protect list also calls connectivity a strength for many; the case for a site-wide "
        "reliability problem is thin — confirm the specific floors or spaces before acting."],
    "va_norfolk": [
        "Priority 5 (Restroom repairs and building upkeep): Building Operations comments fall from 51 to 32 (29 "
        "people) and are 31% negative; the priority now rests mainly on the two Under Supported survey items "
        "(building operations and cleanliness), both with low Friction (13 and 12)."],
    "ny_new_york_penn1": [
        "Priority 5 (Meeting-room hookups, guides and booking): the AV / Video Conferencing leg is 5 comments from 4 "
        "people; Meeting & Conference Rooms (12 comments, 12 people) and the Under Supported virtual-collaboration "
        "item carry it."],
}
