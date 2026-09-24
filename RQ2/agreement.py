#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import openpyxl

BASE_DIR = Path(__file__).resolve().parent

ANNOTATORS = {
    "Annotator1": Path("Annotator1/annotation_sheet.xlsx"),
    "Annotator2": Path("Annotator2/annotation_sheet.xlsx"),
}

SCALE = ["Completely Inaccurate", "Mostly Inaccurate", "Partially Accurate",
         "Mostly Accurate", "Completely Accurate"]

COL_SAMPLE_ID = 0
COL_RATING = 1
COL_ORIGINAL_ID = 2


def norm(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def load_sheet(path: Path) -> dict[str, str]:
    ws = openpyxl.load_workbook(path, data_only=True).worksheets[0]
    ratings: dict[str, str] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        sample_id = norm(row[COL_SAMPLE_ID])
        if not sample_id or sample_id.upper() == "EXAMPLE":
            continue
        original_id = norm(row[COL_ORIGINAL_ID])
        if not original_id:
            continue
        if original_id in ratings:
            raise ValueError(f"{path.name}: duplicate original id {original_id}")
        rating = norm(row[COL_RATING])
        if rating not in SCALE:
            raise ValueError(f"{path.name}: {original_id}: bad rating {rating!r}")
        ratings[original_id] = rating
    return ratings


def weights_matrix() -> dict[tuple[str, str], float]:
    q = len(SCALE)
    idx = {c: i for i, c in enumerate(SCALE)}
    return {(x, y): 1.0 - abs(idx[x] - idx[y]) / (q - 1)
            for x in SCALE for y in SCALE}


def gwet_ac2(pairs: list[tuple[str, str]]) -> tuple[float, float]:
    n = len(pairs)
    q = len(SCALE)
    w = weights_matrix()

    po = sum(w[(a, b)] for a, b in pairs) / n

    pi = {}
    for k in SCALE:
        n_a = sum(1 for a, _ in pairs if a == k)
        n_b = sum(1 for _, b in pairs if b == k)
        pi[k] = (n_a / n + n_b / n) / 2.0

    tw = sum(w.values())
    pe = (tw / (q * (q - 1))) * sum(p * (1.0 - p) for p in pi.values())

    return (po - pe) / (1.0 - pe), po


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets-dir", type=Path, default=BASE_DIR)
    ap.add_argument("--csv", type=Path, default=BASE_DIR / "agreement_results.csv")
    args = ap.parse_args()

    data = {}
    for name, rel in ANNOTATORS.items():
        path = args.sheets_dir / rel
        if not path.is_file():
            print(f"missing sheet: {path}", file=sys.stderr)
            return 1
        data[name] = load_sheet(path)

    a, b = ANNOTATORS
    shared = sorted(set(data[a]) & set(data[b]))
    pairs = [(data[a][oid], data[b][oid]) for oid in shared]

    value, po_w = gwet_ac2(pairs)
    n_agree = sum(1 for x, y in pairs if x == y)

    print(f"INTER-RATER RELIABILITY ({len(shared)} overlapping samples)")
    print(f"{'task':<8}{'type':<10}{'q':>3}{'n':>4}{'agr':>5}"
          f"{'weighted agreement':>21}{'coefficient':>14}{'value':>9}")
    print(f"{'Task':<8}{'ordinal':<10}{len(SCALE):>3}{len(pairs):>4}{n_agree:>5}"
          f"{po_w:>21.3f}{'AC2-linear':>14}{value:>+9.3f}")

    row = {
        "task": "Task",
        "type": "ordinal",
        "q": len(SCALE),
        "n": len(pairs),
        "agr": n_agree,
        "weighted_agreement": round(po_w, 4),
        "coefficient": "AC2-linear",
        "value": round(value, 4),
    }
    with args.csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)

    return 0


if __name__ == "__main__":
    sys.exit(main())
