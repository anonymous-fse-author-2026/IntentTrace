from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import openpyxl

BASE_DIR = Path(__file__).resolve().parent

ANNOTATORS = {
    "Annotator1": Path("Annotator1/annotation_sheet.xlsx"),
    "Annotator2": Path("Annotator2/annotation_sheet.xlsx"),
}
MANIFEST = Path("manifest.csv")

DATASETS = ["PAGED", "Industry"]
LLMS = ["gemini", "qwen"]
BANDS = ["small", "medium", "large"]


def norm(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def load_annotated_ids(path: Path) -> set[str]:
    ws = openpyxl.load_workbook(path, data_only=True).worksheets[0]
    ids = set()
    for row in ws.iter_rows(min_row=2, values_only=True):
        sample_id = norm(row[0])
        if not sample_id or sample_id.upper() == "EXAMPLE":
            continue
        original_id = norm(row[2])
        if original_id:
            ids.add(original_id)
    return ids


def load_strata(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as fh:
        strata = {norm(r["sample_id"]): {
            "dataset": norm(r["dataset"]),
            "llm": norm(r["model_run"]).split("-")[0],
            "band": norm(r["band"]),
        } for r in csv.DictReader(fh)}

    for oid, s in strata.items():
        for field, allowed in (("dataset", DATASETS), ("llm", LLMS), ("band", BANDS)):
            if s[field] not in allowed:
                raise ValueError(f"{oid}: unknown {field} {s[field]!r}")
    return strata


def table(ids: list[str], strata: dict[str, dict], title: str) -> None:
    print("\n" + "=" * 74)
    print(f"{title}  (n={len(ids)})")
    print("=" * 74)
    c = Counter((strata[i]["dataset"], strata[i]["llm"], strata[i]["band"])
                for i in ids)
    print(f"{'dataset':<10}{'llm':<9}" + "".join(f"{b:>9}" for b in BANDS)
          + f"{'total':>9}")
    print("-" * 74)
    for ds in DATASETS:
        for llm in LLMS:
            cells = [c[(ds, llm, b)] for b in BANDS]
            print(f"{ds:<10}{llm:<9}" + "".join(f"{v:>9}" for v in cells)
                  + f"{sum(cells):>9}")
        cells = [sum(c[(ds, llm, b)] for llm in LLMS) for b in BANDS]
        print(f"{ds + ' total':<19}" + "".join(f"{v:>9}" for v in cells)
              + f"{sum(cells):>9}")
        print("-" * 74)
    cells = [sum(c[(ds, llm, b)] for ds in DATASETS for llm in LLMS)
             for b in BANDS]
    print(f"{'all':<19}" + "".join(f"{v:>9}" for v in cells)
          + f"{sum(cells):>9}")


def rows_for_csv(ids: list[str], strata: dict[str, dict]) -> list[dict]:
    c = Counter((strata[i]["dataset"], strata[i]["llm"], strata[i]["band"])
                for i in ids)
    return [{"dataset": ds, "llm": llm, "band": b, "n": c[(ds, llm, b)]}
            for ds in DATASETS for llm in LLMS for b in BANDS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets-dir", type=Path, default=BASE_DIR)
    ap.add_argument("--csv", type=Path, default=BASE_DIR / "stratification.csv")
    args = ap.parse_args()

    sets = {}
    for name, rel in ANNOTATORS.items():
        path = args.sheets_dir / rel
        if not path.is_file():
            print(f"missing sheet: {path}", file=sys.stderr)
            return 1
        sets[name] = load_annotated_ids(path)

    strata = load_strata(args.sheets_dir / MANIFEST)

    a, b = ANNOTATORS
    shared = sorted(sets[a] & sets[b])
    unique = sorted(sets[a] | sets[b])

    missing = [i for i in unique if i not in strata]
    if missing:
        print(f"no strata for {missing}", file=sys.stderr)
        return 1

    print("SAMPLE COMPOSITION")
    print("  " + "  ".join(f"{n}={len(s)}" for n, s in sets.items())
          + f"  shared={len(shared)}")
    print("  unique diagrams = "
          + " + ".join(str(len(s)) for s in sets.values())
          + f" - {len(shared)} = {len(unique)}")

    table(unique, strata, "STRATIFICATION OF UNIQUE SAMPLED DIAGRAMS")

    out = rows_for_csv(unique, strata)
    with args.csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
