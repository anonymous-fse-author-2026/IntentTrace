from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLED_ROOT = HERE.parent / "Sampled"
OUT_CSV = HERE / "stratification.csv"

DATASETS = ("Industry", "PAGED")


def main() -> None:
    counts = Counter()
    for dataset in DATASETS:
        mapping = SAMPLED_ROOT / dataset / "mapping.csv"
        if not mapping.is_file():
            print(f"[skip] no mapping.csv for {dataset}")
            continue
        with mapping.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):

                counts[(dataset, row["llm"], row["variant"])] += 1

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "llm", "approach", "n_samples"])
        for key in sorted(counts):
            w.writerow([*key, counts[key]])
        w.writerow(["TOTAL", "", "", sum(counts.values())])

    for key in sorted(counts):
        print(f"{key[0]:<10}{key[1]:<12}{key[2]:<16}{counts[key]:>5}")
    print(f"{'TOTAL':<38}{sum(counts.values()):>5}")
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
