"""
Columns in the raw PulsePoint export that the tool never uses and that identify
or locate a respondent. They are removed from the de-identified copy
(tools/deidentify_raw.py) and dropped again by build_data.py on load, whatever
file it is given. Matching is case-insensitive.
"""
import re

# exact column names (compared case-insensitively, surrounding whitespace ignored)
PII_EXACT = ["IP", "Location (via IP)", "Browser", "Operating system",
             "Date responded", "Referring sites"]
# the optional contact question that asks for a network ID — matched by substring
PII_CONTAINS = ["us domain id"]


def _norm(name) -> str:
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def is_pii_column(name) -> bool:
    n = _norm(name)
    return n in {_norm(x) for x in PII_EXACT} or any(s in n for s in PII_CONTAINS)


def pii_columns(columns) -> list:
    return [c for c in columns if is_pii_column(c)]


def drop_pii_columns(df, notice=print):
    """Return df without PII columns; print a notice listing what was dropped."""
    found = pii_columns(df.columns)
    if found and notice:
        notice(f"[deid] dropping {len(found)} identifying column(s) on load: "
               + ", ".join(str(c)[:40] for c in found))
    return df.drop(columns=found)
