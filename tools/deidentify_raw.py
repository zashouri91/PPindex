"""
Write a de-identified copy of the raw PulsePoint export.

    python tools/deidentify_raw.py [data/raw.xlsx] [data/raw_deid.xlsx]

Removes the columns listed in deid.py (network-ID contact question, IP,
location, browser, OS, response date, referring sites). Everything else is
kept as-is: same sheet name, same remaining columns in the same order, same
cell values — including Respondent, Reached end, Completion time (seconds)
and every survey question. Works on the workbook with openpyxl so cell values
are not re-typed by pandas.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import openpyxl  # noqa: E402
from deid import is_pii_column  # noqa: E402


def deidentify(src: str, dst: str) -> list:
    wb = openpyxl.load_workbook(src)
    ws = wb[wb.sheetnames[0]]
    header = [c.value for c in ws[1]]
    drop_idx = [i + 1 for i, h in enumerate(header) if h is not None and is_pii_column(h)]
    for idx in sorted(drop_idx, reverse=True):  # right to left keeps indexes valid
        ws.delete_cols(idx)
    # only the first (data) sheet is kept
    for name in wb.sheetnames[1:]:
        del wb[name]
    wb.save(dst)
    return [header[i - 1] for i in drop_idx]


def main(argv):
    src = argv[1] if len(argv) > 1 else os.path.join(ROOT, "data", "raw.xlsx")
    dst = argv[2] if len(argv) > 2 else os.path.join(ROOT, "data", "raw_deid.xlsx")
    dropped = deidentify(src, dst)
    print(f"wrote {dst}; removed {len(dropped)} column(s):")
    for c in dropped:
        print("  -", str(c)[:60])


if __name__ == "__main__":
    main(sys.argv)
