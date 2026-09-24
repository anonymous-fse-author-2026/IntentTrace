import os
from statistics import mean

import table_common as T
import rq1_data as D

SIG = D.SIG
RC = D.RC
R = D.R
BASE = os.path.join(T.BASE_DIR, "rq1_cross_model")

SLOW_MODEL = "qwen"
FAST_MODEL = "gemini"
FAST_COLUMN = R.round_col(1)
SLOW_COLUMNS = [R.round_col(1), R.round_col(R.max_round())]


def per_diagram(column, dataset, model):
    runs = R.RUNS
    if R.is_round(column) and model in R.SINGLE_RUN_MODELS:
        runs = [R.SINGLE_RUN_MODELS[model]]
    return R.collect_per_diagram(column, dataset, model, runs)


def collapse(scores):
    by_case = {}
    for (_run, case), vals in scores.items():
        by_case.setdefault(case, []).append(vals)
    return {case: {m: mean(v[m] for v in vals) for m in SIG.SIG_METRICS}
            for case, vals in by_case.items()}


def compare(dataset):
    fast = collapse(per_diagram(FAST_COLUMN, dataset, FAST_MODEL))
    block = []
    for slow_column in SLOW_COLUMNS:
        slow = collapse(per_diagram(slow_column, dataset, SLOW_MODEL))
        common = sorted(set(slow) & set(fast))
        for metric in SIG.SIG_METRICS:
            rec = SIG.paired_test([slow[c][metric] for c in common],
                                  [fast[c][metric] for c in common])
            rec.update(dataset=dataset, column=slow_column, metric=metric)
            block.append(rec)

    for rec, p in zip(block, SIG.benjamini_hochberg(
            [r["p_value"] for r in block])):
        rec["p_value_bh"] = p
        rec["significant_bh_0.05"] = bool(p == p and p < 0.05)
    return block


def head(column):
    return "%s %s vs %s %s" % (RC.model_label(SLOW_MODEL),
                               R.column_label(column),
                               RC.model_label(FAST_MODEL),
                               R.column_label(FAST_COLUMN))


def main():
    summary = D.summary()
    by_key = {}
    for dataset in D.DATASETS:
        for rec in compare(dataset):
            by_key[(dataset, rec["column"], rec["metric"])] = rec

    header = ["dataset", "comparison", "metric", "n_pairs",
              "qwen_mean", "qwen_sd", "gemini_mean", "gemini_sd",
              "mean_diff", "p_value", "p_value_bh", "significant_bh_0.05",
              "A12", "A12_effect"]
    csv_rows = []
    for dataset in D.DATASETS:
        for column in SLOW_COLUMNS:
            for metric in SIG.SIG_METRICS:
                rec = by_key.get((dataset, column, metric))
                if rec is None:
                    continue
                csv_rows.append([
                    dataset, head(column),
                    SIG.SIG_METRIC_LABEL[metric], rec["n_pairs"],
                    "%.4f" % rec["mean_A"],
                    _sd(summary, dataset, SLOW_MODEL, column, metric),
                    "%.4f" % rec["mean_B"],
                    _sd(summary, dataset, FAST_MODEL, FAST_COLUMN, metric),
                    "%.4f" % rec["mean_diff"],
                    "%.6g" % rec["p_value"], "%.6g" % rec["p_value_bh"],
                    rec["significant_bh_0.05"],
                    "%.4f" % rec["A12"], rec["A12_effect"]])

    return [T.write_csv(BASE + ".csv", header, csv_rows)]


def _sd(summary, dataset, model, column, metric):
    row = summary.get((dataset, model, column))
    if row is None:
        return ""
    sd = T.sd_of(row, "avg_" + metric)
    return "" if sd is None else "%.4f" % sd


if __name__ == "__main__":
    for p in main():
        print(f"Wrote {p}")
