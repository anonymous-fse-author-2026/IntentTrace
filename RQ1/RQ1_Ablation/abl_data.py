import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import ablation_results_reader as A

RC = A.RC
SIG = A.SIG

DATASETS = A.DATASETS
MODELS = A.MODELS
BASELINE = A.BASELINE
ROUND1 = A.ROUND1
ABLATIONS = A.ABLATIONS
COLUMNS = [BASELINE] + ABLATIONS + [ROUND1]
BLOCKS = A.QUALITY_BLOCKS


def summary():
    out = {}
    for dataset in DATASETS:
        for model in MODELS:
            for column in COLUMNS:
                _scored, agg = A.collect_column(
                    column, dataset, model,
                    carry_forward=column in ABLATIONS,
                    all_runs_cost=column == ROUND1)
                if agg is not None:
                    out[(dataset, model, column)] = agg
    return out


def significance(kind):
    records = []
    for dataset in DATASETS:
        for model in MODELS:
            per_diagram = {
                c: A.collect_per_diagram(c, dataset, model)
                for c in (BASELINE, ROUND1)}
            for ablation in ABLATIONS:
                per_diagram[ablation] = A.collect_per_diagram(
                    ablation, dataset, model, carry_forward=True)
            SIG.run_block(
                dataset, model, A.SIG_TABLE_PAIRS, per_diagram,
                label_fn=lambda c: A.COLUMN_LABEL[c],
                model_label_fn=RC.model_label,
                out_records=records,
                metrics=SIG.SIG_METRIC_FAMILIES[kind])
    return records, A.SIG_TABLE_PAIRS


def stage_costs():
    return {(d, m): A.stage_stats(d, m) for d in DATASETS for m in MODELS}


def spec_summary():
    from statistics import mean, pstdev

    out = {}
    for dataset in DATASETS:
        mapping = A.spec_map(dataset)
        for model in MODELS:
            for column in COLUMNS:
                carry = column in ABLATIONS
                per_spec = A.collect_per_spec(column, dataset, model, mapping,
                                              carry_forward=carry)
                row = A.aggregate_specs(per_spec, column, dataset, model)
                if row is None:
                    continue

                runs = A.runs_for(column, model)
                if len(runs) > 1:
                    per_run = []
                    for run in runs:
                        ps = A.collect_per_spec(
                            column, dataset, model, mapping,
                            carry_forward=carry, runs=[run])
                        if ps:
                            per_run.append(ps)
                    for key in A.SCORE_KEYS:
                        vals = [mean(p[s][key] for s in sorted(p))
                                for p in per_run]
                        if len(vals) > 1:
                            row[f"avg_{key}_sd"] = pstdev(vals)
                    vals = [sum(p[s]["struct_or_logic"] for s in sorted(p))
                            for p in per_run]
                    if len(vals) > 1:
                        row["exp_specs_struct_or_logic_sd"] = pstdev(vals)
                out[(dataset, model, column)] = row
    return out


def spec_significance(kind="neuro"):
    records = []
    for dataset in DATASETS:
        mapping = A.spec_map(dataset)
        for model in MODELS:
            per_spec = {}
            for column in COLUMNS:
                ps = A.collect_per_spec(column, dataset, model, mapping,
                                        carry_forward=column in ABLATIONS)
                per_spec[column] = {spec: vals for spec, vals in ps.items()}
            block = SIG.run_block(
                dataset, model, A.SIG_TABLE_PAIRS, per_spec,
                label_fn=lambda c: A.COLUMN_LABEL[c],
                model_label_fn=RC.model_label,
                metrics=SIG.SIG_METRIC_FAMILIES[kind])
            for rec in block:
                runs = sorted(
                    {str(r) for col in (rec["group_A"], rec["group_B"])
                     for r in A.runs_for(col, model)},
                    key=lambda x: int(x) if x.isdigit() else x)
                rec["runs"] = ",".join(runs)
                rec["n_runs"] = len(runs)
            records.extend(block)
    return records, A.SIG_TABLE_PAIRS
