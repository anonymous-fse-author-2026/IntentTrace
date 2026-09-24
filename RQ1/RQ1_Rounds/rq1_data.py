import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import results_reader as R

RC = R.RC
SIG = R.SIG

DATASETS = R.DATASETS
MODELS = R.MODELS
BASELINE = R.BASELINE


def columns():
    return [BASELINE] + [R.round_col(n) for n in range(1, R.max_round() + 1)]


def summary():
    out = {}
    for dataset in DATASETS:
        for model in MODELS:
            for column in columns():
                score_runs = R.RUNS
                if R.is_round(column) and model in R.SINGLE_RUN_MODELS:
                    score_runs = [R.SINGLE_RUN_MODELS[model]]

                found = [R.collect_run(column, dataset, model, r)
                         for r in score_runs]
                found = [r for r in found if r is not None]
                if not found:
                    continue

                extra = []
                if (column in R.COST_ALL_RUNS_COLUMNS
                        and model in R.SINGLE_RUN_MODELS):
                    seen = {r["run"] for r in found}
                    extra = [R.collect_run(column, dataset, model, r)
                             for r in R.RUNS if r not in seen]
                    extra = [r for r in extra if r is not None]

                scored = [r for r in found if r["n_evaluated"] > 0] or found
                out[(dataset, model, column)] = R.aggregate_runs(
                    scored, found + extra)
    return out


def significance(kind):
    cols = columns()
    consecutive = R.significance_pairs(cols)[0]
    records = []
    for dataset in DATASETS:
        for model in MODELS:
            per_diagram = {}
            for column in cols:
                runs = R.RUNS
                if R.is_round(column) and model in R.SINGLE_RUN_MODELS:
                    runs = [R.SINGLE_RUN_MODELS[model]]
                per_diagram[column] = R.collect_per_diagram(
                    column, dataset, model, runs)
            SIG.run_block(
                dataset, model, consecutive, per_diagram,
                label_fn=R.column_label, model_label_fn=RC.model_label,
                out_records=records,
                metrics=SIG.SIG_METRIC_FAMILIES[kind])
    return records, consecutive
