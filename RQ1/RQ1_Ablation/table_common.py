import csv
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RQ1_DIR = os.path.dirname(BASE_DIR)
if RQ1_DIR not in sys.path:
    sys.path.insert(0, RQ1_DIR)


def has_spread(row):
    try:
        return int(float(row.get("n_runs", 0) or 0)) > 1
    except (TypeError, ValueError):
        return False


def sd_of(row, field):
    if not has_spread(row):
        return None
    val = row.get(field + "_sd")
    return None if val is None else float(val)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path
