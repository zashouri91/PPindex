"""Survey waves (rounds). Each wave is built and reported on its own.

Round 1 is the original Tableau-reconciled set: 10 in-scope sites, benchmark = those
10 sites pooled ("company"). Its numbers are locked by tests/test_baseline.py.

Round 2 is the same questionnaire at a different set of sites. Every round 2 site is
in scope, and its benchmark is the round 2 sites pooled — round 2 sites are never
ranked against round 1 numbers (see the combined portfolio card for side-by-side).

`python build_data.py --wave round2` builds a wave into its own data folder.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

ROUND1_SITES = ["CA-Woodland Hills", "FL-Miami", "FL-Tampa", "GA-Atlanta",
                "IL-Chicago-Willis Tower", "IN-Indianapolis", "NY-New York (Penn1)",
                "OH-Mason", "VA-Norfolk", "VA-Richmond"]

# Round 2 export names sites by street address. Short names follow the round 1 style
# (state-city, with a suffix only where a city has two sites). Matched on the text
# before the first comma, case-insensitive.
ROUND2_SITE_NAMES = {
    "CT-WALLINGFORD": "CT-Wallingford",
    "NY-LATHAM": "NY-Latham",
    "KY-LOUISVILLE": "KY-Louisville",
    "MO-ST. LOUIS": "MO-St. Louis",
    "MN-MENDOTA HEIGHTS": "MN-Mendota Heights",
    "GA-COLUMBUS": "GA-Columbus",
    "FL-LAKE MARY": "FL-Lake Mary",
    "WI-WAUKESHA": "WI-Waukesha",
    "NJ-ISELIN": "NJ-Iselin",
    "IL-CHICAGO": "IL-Chicago-Bryn Mawr",          # 8600 W Bryn Mawr Ave
    "NH-MANCHESTER": "NH-Manchester",
    "TN-NASHVILLE": "TN-Nashville",
    "CA-COSTA MESA": "CA-Costa Mesa",
    "MD-HANOVER": "MD-Hanover",
    "NC-DURHAM": "NC-Durham",
    "NY-LAKE SUCCESS": "NY-Lake Success",
    "DE-WILMINGTON": "DE-Wilmington",
    "MA-WOBURN": "MA-Woburn",
    "VA-ASHBURN": "VA-Ashburn",
    "CA-WALNUT CREEK": "CA-Walnut Creek",
    "LA-HARVEY": "LA-Harvey",
    "CT-ROCKY HILL": "CT-Rocky Hill",
    "WA-SEATTLE": "WA-Seattle",
    "NC-WINSTON-SALEM": "NC-Winston-Salem",
    "IA-URBANDALE": "IA-Urbandale",
    "KS-OVERLAND PARK": "KS-Overland Park",
    "DC-WASHINGTON": "DC-Washington",
    "PA-SEVEN FIELDS": "PA-Seven Fields",
}
# Two Las Vegas sites: told apart by street.
ROUND2_STREET_NAMES = {
    "9133 W RUSSELL RD": "NV-Las Vegas-Russell",
    "3634 S MARYLAND PKWY": "NV-Las Vegas-Maryland Pkwy",
}

# Sites below this many completed responses are flagged "small sample" on their card.
LOW_N = 30

WAVES = {
    "round1": {
        "key": "round1", "label": "Round 1",
        "data_dir": os.path.join(HERE, "data"),
        "raw": os.path.join(HERE, "data", "raw_deid.xlsx"),
        "sites": ROUND1_SITES,
        "bench_noun": "company",                 # "Below the company average"
        "bench_title": "Company",
        "scope_phrase": "10 in-scope sites",
        "company_row": "Company (10 sites)",
        "card_title": "10 PulsePoints Report Card",
    },
    "round2": {
        "key": "round2", "label": "Round 2",
        "data_dir": os.path.join(HERE, "data", "round2"),
        "raw": os.path.join(HERE, "data", "round2", "raw_deid.xlsx"),
        "sites": None,                           # every site in the export is in scope
        "bench_noun": "round 2",                 # "Below the round 2 average"
        "bench_title": "Round 2 average",
        "scope_phrase": None,                    # filled at build: "30 round 2 sites"
        "company_row": None,                     # filled at build: "Round 2 (30 sites)"
        "card_title": "Round 2 PulsePoints Report Card",
    },
}


def normalise_round2_site(raw):
    """Map a round 2 address-style site name to its short name. Unknown names pass
    through unchanged (and are reported by build_data)."""
    if raw is None or (isinstance(raw, float) and raw != raw):
        return raw
    s = str(raw).strip()
    up = s.upper()
    for street, name in ROUND2_STREET_NAMES.items():
        if street in up:
            return name
    head = up.split(",")[0].strip()
    return ROUND2_SITE_NAMES.get(head, s)


def wave_meta(wave, sites):
    """The wave fields written into meta.json for reports.py."""
    w = dict(WAVES[wave])
    n = len(sites)
    if w["scope_phrase"] is None:
        w["scope_phrase"] = f"{n} {w['label'].lower()} sites"
    if w["company_row"] is None:
        w["company_row"] = f"{w['label']} ({n} sites)"
    return {"key": w["key"], "label": w["label"], "bench_noun": w["bench_noun"],
            "bench_title": w["bench_title"], "scope_phrase": w["scope_phrase"],
            "company_row": w["company_row"], "card_title": w["card_title"], "low_n": LOW_N}
