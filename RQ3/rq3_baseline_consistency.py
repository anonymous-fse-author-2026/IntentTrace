import os
import sys
from statistics import mean, stdev

from scipy.stats import spearmanr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
RQ1_DIR = os.path.join(REPO_ROOT, "RQ1")
if RQ1_DIR not in sys.path:
    sys.path.insert(0, RQ1_DIR)

import report_common as RC

RESULTS_ROOT = os.path.join(REPO_ROOT, "Results")
APPROACH = "IntentTrace"
ROUND = "Round0"
BASE = os.path.join(BASE_DIR, "rq3_baseline_consistency")

DATASETS = ["Industry", "PAGED"]
MODELS = ["gemini", "qwen"]
RUNS = [1, 2, 3, 4, 5]

MEASURES = {
    "spec": ("recall", "precision"),
    "abscon": ("abscon_recall", "abscon_precision"),
    "ladex": ("ladex_recall", "ladex_precision"),
}
MEASURE_LABEL = {"spec": "Spec-alignment", "abscon": "AbsCon", "ladex": "LADEX"}
DETERMINISTIC = {"abscon", "ladex"}

COMPARISONS = [("abscon", "ladex"), ("spec", "abscon"), ("spec", "ladex")]
METRICS = [("Recall", 0), ("Precision", 1)]

HEADER = ["comparison", "dataset", "llm", "metric",
          "n_diagrams", "n_runs", "rho", "rho_sd", "p_max"]


def load_run(dataset, model, run):
    group_dir = os.path.join(RESULTS_ROOT, ROUND, APPROACH,
                             dataset, f"{model}-{run}")
    if not os.path.isdir(group_dir):
        return {}
    out = {}
    for case in sorted(os.listdir(group_dir)):
        case_dir = os.path.join(group_dir, case)
        if not os.path.isdir(case_dir):
            continue
        data = RC.load_json(os.path.join(case_dir, f"{case}.metrics"))
        pr = (data or {}).get("precision_recall")
        if not pr:
            continue
        ab_p, ab_r = RC.baseline_pair(data, "abscon")
        la_p, la_r = RC.baseline_pair(data, "ladex")
        out[case] = {
            "recall": RC.pr_recall(pr),
            "precision": RC.pr_precision(pr),
            "abscon_recall": ab_r,
            "abscon_precision": ab_p,
            "ladex_recall": la_r,
            "ladex_precision": la_p,
        }
    return out


def is_deterministic(x_key, y_key):
    return x_key in DETERMINISTIC and y_key in DETERMINISTIC


def comparison_label(x_key, y_key):
    return f"{MEASURE_LABEL[x_key]} vs {MEASURE_LABEL[y_key]}"


def correlate(runs, dataset, model, x_key, y_key, sel):
    x_field = MEASURES[x_key][sel]
    y_field = MEASURES[y_key][sel]

    rhos, pvals, n_used = [], [], 0
    for run in RUNS:
        values = runs.get((dataset, model, run)) or {}
        if len(values) < 2:
            continue
        cases = sorted(values)
        rho, pval = spearmanr([values[c][x_field] for c in cases],
                              [values[c][y_field] for c in cases])
        rhos.append(float(rho))
        pvals.append(float(pval))
        n_used = len(cases)
        if is_deterministic(x_key, y_key):
            break

    if not rhos:
        return None
    return {
        "n_diagrams": n_used,
        "n_runs": len(rhos),
        "rho": mean(rhos),
        "rho_sd": stdev(rhos) if len(rhos) > 1 else 0.0,
        "p_max": max(pvals),
    }


def summary():
    runs = {(dataset, model, run): load_run(dataset, model, run)
            for dataset in DATASETS for model in MODELS for run in RUNS}
    out = {}
    for x_key, y_key in COMPARISONS:
        for metric, sel in METRICS:
            for dataset in DATASETS:
                for model in MODELS:
                    rec = correlate(runs, dataset, model, x_key, y_key, sel)
                    if rec is not None:
                        out[(x_key, y_key, dataset, model, metric)] = rec
    return out


def csv_rows(results):
    rows = []
    for x_key, y_key in COMPARISONS:
        for dataset in DATASETS:
            for model in MODELS:
                for metric, _sel in METRICS:
                    rec = results.get((x_key, y_key, dataset, model, metric))
                    if rec is None:
                        continue
                    rows.append([
                        comparison_label(x_key, y_key),
                        dataset,
                        RC.model_label(model),
                        metric,
                        rec["n_diagrams"],
                        rec["n_runs"],
                        "%.4f" % rec["rho"],
                        "%.4f" % rec["rho_sd"],
                        "%.3e" % rec["p_max"],
                    ])
    return rows


def write_csv(path, header, rows):
    import csv

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def main():
    rows = csv_rows(summary())

    width = max(len(r[0]) for r in rows)
    print(f"{'comparison':<{width}} {'dataset':<9}{'llm':<8}{'metric':<10}"
          f"{'n':>5}{'runs':>6}{'rho':>8}{'SD':>7}{'p_max':>12}")
    for row in rows:
        print(f"{row[0]:<{width}} {row[1]:<9}{row[2]:<8}{row[3]:<10}"
              f"{row[4]:>5}{row[5]:>6}{float(row[6]):>8.2f}"
              f"{float(row[7]):>7.2f}{row[8]:>12}")

    return [write_csv(BASE + ".csv", HEADER, rows)]


if __name__ == "__main__":
    for p in main():
        print(f"Wrote {p}")
