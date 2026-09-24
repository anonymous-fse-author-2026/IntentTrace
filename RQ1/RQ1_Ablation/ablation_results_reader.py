import csv
import os
import sys
from collections import defaultdict
from statistics import mean

RQ1_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(RQ1_DIR)
if RQ1_DIR not in sys.path:
    sys.path.insert(0, RQ1_DIR)

import report_common as RC
import sig_common as SIG

RESULTS_ROOT = os.path.join(REPO_ROOT, "Results")

APPROACH_DIR = {
    "Refinement_Using_Only_Struct": "Model-Internal",
    "Refinement_Using_Only_Element": "Element-Alignment",
    "Refinement_Using_LLM": "Direct-LLM",
}
FULL_APPROACH = "IntentTrace"

MODELS = ["gemini", "qwen"]
DATASETS = ["Industry", "PAGED"]
RUNS = [1, 2, 3, 4, 5]
BASELINE = "baseline"
ROUND1 = "round1"

SINGLE_RUN_MODELS = {"gemini": 1}

ABLATIONS = [
    "Refinement_Using_Only_Struct",
    "Refinement_Using_Only_Element",
    "Refinement_Using_LLM",
]

COLUMN_LABEL = {
    BASELINE: "No Refinement",
    ROUND1: "IntentTrace",
    "Refinement_Using_Only_Struct": "Model-Internal",
    "Refinement_Using_Only_Element": "Element Alignment",
    "Refinement_Using_LLM": "Direct LLM",
}

QUALITY_BLOCKS = [("Industry", "gemini"), ("Industry", "qwen"),
                  ("PAGED", "gemini"), ("PAGED", "qwen")]

QUALITY_TABLES = {
    "neuro": {"fields": ("avg_precision", "avg_recall", "avg_f1")},
    "abscon": {"fields": ("avg_abscon_precision", "avg_abscon_recall",
                          "avg_abscon_f1")},
    "ladex": {"fields": ("avg_ladex_precision", "avg_ladex_recall",
                         "avg_ladex_f1")},
}
SIG_TABLE_PAIRS = ([(a, BASELINE) for a in ABLATIONS]
                   + [(ROUND1, a) for a in ABLATIONS])

COST_BUCKETS = ("match", "refine")

SKIP_LABELS = {"trace_builder", "root_cause_analysis"}
ELEMENT_LABELS = {"element_mapping", "var_actor_validation"}

DIAG_LABELS = {
    "Refinement_Using_Only_Struct": frozenset(),
    "Refinement_Using_Only_Element": frozenset(ELEMENT_LABELS),
    "Refinement_Using_LLM": frozenset({"llm_semantic_feedback"}),
    ROUND1: frozenset(ELEMENT_LABELS | {"trace_builder"}),
}

REFINE_TABLE_COLUMNS = ABLATIONS + [ROUND1]

STAGE_LABELS = [("diag", "Diagnosis"), ("refine", "Refinement"),
                ("combined", "Combined")]

MAPPING_CSV = os.path.join(REPO_ROOT, "Datasets", "Sampled", "{dataset}",
                           "mapping.csv")

SCORE_KEYS = ["precision", "recall", "f1"] + [
    f"{tool}_{m}" for tool in ("abscon", "ladex")
    for m in ("precision", "recall", "f1")]


def results_dir(round_no, approach, dataset, model, run):
    return os.path.join(RESULTS_ROOT, f"Round{round_no}", approach,
                        dataset, f"{model}-{run}")


def diag_dir_for(column, dataset, model, run):
    if column == ROUND1:
        return results_dir(0, FULL_APPROACH, dataset, model, run)
    return results_dir(0, APPROACH_DIR[column], dataset, model, run)


def group_dir_for(column, dataset, model, run):
    if column == BASELINE:
        return results_dir(0, FULL_APPROACH, dataset, model, run)
    if column == ROUND1:
        return results_dir(1, FULL_APPROACH, dataset, model, run)
    return results_dir(1, APPROACH_DIR[column], dataset, model, run)


def column_label(column):
    return COLUMN_LABEL.get(column, column)


def runs_for(column, model):
    if column == BASELINE or model not in SINGLE_RUN_MODELS:
        return list(RUNS)
    return [SINGLE_RUN_MODELS[model]]


def split_cost(case_dir, case):
    buckets = RC.empty_cost(COST_BUCKETS)
    for suffix in ("cost", "cost_refinement"):
        data = RC.load_json(os.path.join(case_dir, f"{case}.{suffix}"))
        if data is None:
            continue
        for call in data.get("calls") or []:
            label = str(call.get("label") or "")
            if label in SKIP_LABELS:
                continue
            RC.add_call(buckets["refine" if label.startswith("refine")
                                else "match"], call)
    return buckets


def was_refined(case_dir, case):
    for suffix in ("cost_refinement", "cost"):
        data = RC.load_json(os.path.join(case_dir, f"{case}.{suffix}"))
        if data is None:
            continue
        for call in data.get("calls") or []:
            if str(call.get("label") or "").startswith("refine"):
                return True
    return False


def refined_case_ids(column, dataset, model):
    found = set()
    for run in runs_for(column, model):
        d = group_dir_for(column, dataset, model, run)
        if not os.path.isdir(d):
            continue
        for case in os.listdir(d):
            case_dir = os.path.join(d, case)
            if os.path.isdir(case_dir) and was_refined(case_dir, case):
                found.add(case)
    return found


def f1_of(precision, recall):
    total = precision + recall
    return (2 * precision * recall / total) if total else 0.0


def baseline_pair(data, tool):
    p, r = RC.baseline_pair(data, tool)
    return p, r, f1_of(p, r)


def read_case(case_dir, case):
    data = RC.load_json(os.path.join(case_dir, f"{case}.metrics"))
    if data is None:
        return None
    s_err, l_err, n_err = RC.count_feedback_tags(
        os.path.join(case_dir, f"{case}.feedback"))
    pr = data.get("precision_recall") or {}
    sc = data.get("structural_conformance") or {}
    lc = data.get("logical_sanity_conformance") or {}
    bs = data.get("baseline_scores") or {}
    zeroed = RC.is_zeroed(data)
    ab_p, ab_r, ab_f1 = baseline_pair(data, "abscon")
    la_p, la_r, la_f1 = baseline_pair(data, "ladex")

    stale = bool(bs) and not zeroed and not any((ab_p, ab_r, la_p, la_r))
    return {
        "struct": s_err, "logic": l_err, "neuro": n_err,
        "evaluated": bool(pr),
        "zeroed": zeroed,
        "precision": RC.pr_precision(pr),
        "recall": RC.pr_recall(pr),
        "f1": RC.pr_f1(pr),
        "struct_ratio": sc.get("ratio"),
        "logic_ratio": lc.get("ratio"),
        "has_baseline": bool(bs),
        "baseline_stale": stale,
        "abscon_precision": ab_p,
        "abscon_recall": ab_r,
        "abscon_f1": ab_f1,
        "ladex_precision": la_p,
        "ladex_recall": la_r,
        "ladex_f1": la_f1,
    }


def score_case(case_dir, base_dir, case, carry_forward):
    if carry_forward and not was_refined(case_dir, case):
        cand = os.path.join(base_dir, case)
        if os.path.isdir(cand):
            return read_case(cand, case), True

    vals = read_case(case_dir, case)
    if vals is None or not vals["evaluated"]:
        cand = os.path.join(base_dir, case)
        if os.path.isdir(cand):
            base_vals = read_case(cand, case)
            if base_vals is not None and base_vals["evaluated"]:
                return base_vals, True
    return vals, False


def collect_run(column, dataset, model, run, restrict=None,
                carry_forward=False):
    group_dir = group_dir_for(column, dataset, model, run)
    if not os.path.isdir(group_dir):
        return None
    base_dir = group_dir_for(BASELINE, dataset, model, run)

    n_cases = n_zeroed = n_missing_pr = n_evaluated = n_carried = 0
    struct_total = logic_total = neuro_total = 0
    struct_diag = logic_diag = neuro_diag = 0
    sl_union = 0
    with_issue = without_issue = 0
    precisions, recalls, f1s = [], [], []
    ab_prec, ab_rec, ab_f1s = [], [], []
    la_prec, la_rec, la_f1s = [], [], []
    n_no_baseline = n_stale_baseline = 0
    abscon_perfect = ladex_perfect = 0
    struct_ratios, logic_ratios = [], []
    cost = RC.empty_cost(COST_BUCKETS, diagrams=True)

    for case in sorted(os.listdir(group_dir)):
        if restrict is not None and case not in restrict:
            continue
        case_dir = os.path.join(group_dir, case)
        if not os.path.isdir(case_dir):
            continue

        vals, carried = score_case(case_dir, base_dir, case, carry_forward)
        n_carried += int(carried)
        if vals is None:
            continue

        n_cases += 1
        struct_total += vals["struct"]
        logic_total += vals["logic"]
        neuro_total += vals["neuro"]
        struct_diag += vals["struct"] > 0
        logic_diag += vals["logic"] > 0
        neuro_diag += vals["neuro"] > 0
        sl_union += (vals["struct"] > 0 or vals["logic"] > 0)
        if vals["struct"] + vals["logic"] + vals["neuro"] > 0:
            with_issue += 1
        else:
            without_issue += 1

        if vals["evaluated"]:
            n_evaluated += 1
        else:
            n_missing_pr += 1
        n_zeroed += vals["zeroed"]
        precisions.append(vals["precision"])
        recalls.append(vals["recall"])
        f1s.append(vals["f1"])

        ab_prec.append(vals["abscon_precision"])
        ab_rec.append(vals["abscon_recall"])
        ab_f1s.append(vals["abscon_f1"])
        la_prec.append(vals["ladex_precision"])
        la_rec.append(vals["ladex_recall"])
        la_f1s.append(vals["ladex_f1"])
        n_no_baseline += not vals["has_baseline"]
        n_stale_baseline += vals["baseline_stale"]
        abscon_perfect += (vals["abscon_precision"] == 1.0
                           and vals["abscon_recall"] == 1.0)
        ladex_perfect += (vals["ladex_precision"] == 1.0
                          and vals["ladex_recall"] == 1.0)
        if vals["struct_ratio"] is not None:
            struct_ratios.append(vals["struct_ratio"])
        if vals["logic_ratio"] is not None:
            logic_ratios.append(vals["logic_ratio"])

        RC.add_run_cost(cost, split_cost(case_dir, case))

    if n_cases == 0:
        return None

    def avg(xs):
        return mean(xs) if xs else 0.0

    return {
        "column": column, "dataset": dataset, "model": model, "run": run,
        "n_cases": n_cases,
        "n_evaluated": n_evaluated,
        "n_zeroed": n_zeroed,
        "n_missing_pr": n_missing_pr,
        "n_carried_from_baseline": n_carried,
        "n_diagrams_with_issue": with_issue,
        "n_diagrams_without_issue": without_issue,
        "total_structural_errors": struct_total,
        "avg_structural_errors": struct_total / n_cases,
        "n_diagrams_structural_issue": struct_diag,
        "total_logical_errors": logic_total,
        "avg_logical_errors": logic_total / n_cases,
        "n_diagrams_logical_issue": logic_diag,
        "total_neuro_errors": neuro_total,
        "avg_neuro_errors": neuro_total / n_cases,
        "n_diagrams_neuro_issue": neuro_diag,
        "n_diagrams_struct_or_logic": sl_union,
        "avg_structural_conformance": avg(struct_ratios),
        "avg_logical_conformance": avg(logic_ratios),
        "avg_precision": avg(precisions),
        "avg_recall": avg(recalls),
        "avg_f1": avg(f1s),
        "avg_abscon_precision": avg(ab_prec),
        "avg_abscon_recall": avg(ab_rec),
        "avg_abscon_f1": avg(ab_f1s),
        "avg_ladex_precision": avg(la_prec),
        "avg_ladex_recall": avg(la_rec),
        "avg_ladex_f1": avg(la_f1s),
        "n_abscon_perfect": abscon_perfect,
        "n_ladex_perfect": ladex_perfect,
        "n_missing_baseline": n_no_baseline,
        "n_stale_baseline": n_stale_baseline,
        "_cost": cost,
    }


AVG_FIELDS = [
    "n_cases", "n_evaluated", "n_zeroed", "n_missing_pr",
    "n_carried_from_baseline",
    "n_diagrams_with_issue", "n_diagrams_without_issue",
    "total_structural_errors", "avg_structural_errors",
    "n_diagrams_structural_issue",
    "total_logical_errors", "avg_logical_errors", "n_diagrams_logical_issue",
    "total_neuro_errors", "avg_neuro_errors", "n_diagrams_neuro_issue",
    "n_diagrams_struct_or_logic",
    "avg_structural_conformance", "avg_logical_conformance",
    "avg_precision", "avg_recall", "avg_f1",
    "avg_abscon_precision", "avg_abscon_recall", "avg_abscon_f1",
    "avg_ladex_precision", "avg_ladex_recall", "avg_ladex_f1",
    "n_abscon_perfect", "n_ladex_perfect",
    "n_missing_baseline", "n_stale_baseline",
]


def aggregate_runs(score_rows, cost_rows):
    out = RC.aggregate_runs(score_rows, cost_rows, AVG_FIELDS, COST_BUCKETS,
                            ("dataset", "model", "column"), with_usd=False)
    for kind in RC.TOKEN_KINDS:
        out[f"round_{kind}_tokens"] = (out[f"match_{kind}_tokens"]
                                       + out[f"refine_{kind}_tokens"])
        out[f"round_{kind}_per_diagram"] = (out[f"match_{kind}_per_diagram"]
                                            + out[f"refine_{kind}_per_diagram"])
    return out


def collect_column(column, dataset, model, restrict=None, carry_forward=False,
                   all_runs_cost=False):
    score_run_ids = runs_for(column, model)
    found = [collect_run(column, dataset, model, r, restrict, carry_forward)
             for r in score_run_ids]
    found = [r for r in found if r is not None]
    if not found:
        return [], None

    cost_rows = found
    if all_runs_cost and model in SINGLE_RUN_MODELS:
        extra = [collect_run(column, dataset, model, r, restrict, carry_forward)
                 for r in RUNS if r not in set(score_run_ids)]
        cost_rows = found + [r for r in extra if r is not None]

    scored = [r for r in found if r["n_evaluated"] > 0] or found
    return scored, aggregate_runs(scored, cost_rows)


def collect_per_diagram(column, dataset, model, carry_forward=False):
    values = {}
    for run in runs_for(column, model):
        group_dir = group_dir_for(column, dataset, model, run)
        if not os.path.isdir(group_dir):
            continue
        base_dir = group_dir_for(BASELINE, dataset, model, run)
        for case in sorted(os.listdir(group_dir)):
            case_dir = os.path.join(group_dir, case)
            if not os.path.isdir(case_dir):
                continue
            vals, _ = score_case(case_dir, base_dir, case, carry_forward)
            if vals is None or not vals["evaluated"]:
                continue
            values[(str(run), case)] = {
                k: vals[k] for k in SCORE_KEYS}
    return values


def _new_acc():
    return dict(input=0, output=0, reasoning=0, total=0, calls=0,
                seconds=0.0, usd=0.0, n=0, priced=0, timed=0)


def _billable(call):
    return not (call.get("error") and call.get("total_tokens") is None
                and call.get("prompt_tokens") is None)


def _add_call(acc, call):
    prompt = call.get("prompt_tokens") or 0
    completion = call.get("completion_tokens") or 0
    reasoning = call.get("reasoning_tokens") or 0
    acc["input"] += prompt
    acc["output"] += max(completion - reasoning, 0)
    acc["reasoning"] += reasoning
    acc["total"] += call.get("total_tokens") or (prompt + completion)
    acc["calls"] += 1
    if call.get("duration_seconds") is not None:
        acc["seconds"] += call["duration_seconds"]
        acc["timed"] += 1
    if call.get("cost") is not None:
        acc["usd"] += call["cost"]
        acc["priced"] += 1


def diag_call_stats(column, dataset, model):
    labels = DIAG_LABELS[column]
    per_run = []
    for run in RUNS:
        group_dir = diag_dir_for(column, dataset, model, run)
        if not os.path.isdir(group_dir):
            continue
        acc = _new_acc()
        for case in sorted(os.listdir(group_dir)):
            case_dir = os.path.join(group_dir, case)
            if not os.path.isdir(case_dir):
                continue
            acc["n"] += 1
            if not labels:
                continue
            data = RC.load_json(os.path.join(case_dir, f"{case}.cost"))
            if data is None:
                continue
            for call in data.get("calls") or []:
                if str(call.get("label") or "") in labels and _billable(call):
                    _add_call(acc, call)
        if acc["n"]:
            acc["run"] = run
            per_run.append(acc)

    if not per_run:
        return None

    def avg(fn):
        return mean([fn(a) for a in per_run])
    fully_priced = all(a["priced"] == a["calls"] for a in per_run)
    fully_timed = all(a["timed"] == a["calls"] for a in per_run)
    return {
        "n": avg(lambda a: a["n"]),
        "calls": avg(lambda a: a["calls"]),
        "runs": ",".join(str(a["run"]) for a in per_run),
        **{k: avg(lambda a, k=k: a[k] / a["n"]) for k in
           ("input", "output", "reasoning", "total")},
        "seconds": avg(lambda a: a["seconds"] / a["n"]) if fully_timed else None,
        "usd": avg(lambda a: a["usd"] / a["n"]) if fully_priced else None,
        "seconds_sum": avg(lambda a: a["seconds"]) if fully_timed else None,
        "usd_sum": avg(lambda a: a["usd"]) if fully_priced else None,
    }


def refine_call_stats(column, dataset, model):
    per_run = []
    for run in RUNS:
        group_dir = group_dir_for(column, dataset, model, run)
        if not os.path.isdir(group_dir):
            continue
        acc = _new_acc()
        for case in sorted(os.listdir(group_dir)):
            case_dir = os.path.join(group_dir, case)
            if not os.path.isdir(case_dir):
                continue
            hit = False
            for suffix in ("cost", "cost_refinement"):
                data = RC.load_json(os.path.join(case_dir, f"{case}.{suffix}"))
                if data is None:
                    continue
                for call in data.get("calls") or []:
                    if not str(call.get("label") or "").startswith("refine"):
                        continue
                    hit = True
                    if _billable(call):
                        _add_call(acc, call)
            acc["n"] += hit
        if acc["n"]:
            acc["run"] = run
            per_run.append(acc)

    if not per_run:
        return None

    def avg(fn):
        return mean([fn(a) for a in per_run])

    return {
        "n": avg(lambda a: a["n"]),
        "calls": avg(lambda a: a["calls"]),
        "runs": ",".join(str(a["run"]) for a in per_run),
        **{k: avg(lambda a, k=k: a[k] / a["n"]) for k in
           ("input", "output", "reasoning", "total")},
        "seconds": avg(lambda a: a["seconds"] / a["timed"]) if all(
            a["timed"] for a in per_run) else None,
        "usd": avg(lambda a: a["usd"] / a["priced"]) if all(
            a["priced"] for a in per_run) else None,
        "seconds_sum": avg(lambda a: a["seconds"]) if all(
            a["timed"] for a in per_run) else None,
        "usd_sum": avg(lambda a: a["usd"]) if all(
            a["priced"] for a in per_run) else None,
    }


def combine_stats(diag, refine):
    if diag is None or refine is None:
        return None
    out = {}
    for key in ("input", "output", "reasoning", "total", "seconds", "usd"):
        a, b = diag.get(key), refine.get(key)
        out[key] = None if a is None or b is None else a + b
    out["n"] = refine.get("n")
    out["calls"] = None if diag.get("calls") is None else (
        diag["calls"] + refine.get("calls", 0))
    out["runs"] = refine.get("runs")
    out["seconds_sum"] = out["usd_sum"] = None
    return out


def stage_stats(dataset, model):
    out = {}
    for column in REFINE_TABLE_COLUMNS:
        diag = diag_call_stats(column, dataset, model)
        refine = refine_call_stats(column, dataset, model)
        if diag is None and refine is None:
            continue
        out[column] = {"diag": diag, "refine": refine,
                       "combined": combine_stats(diag, refine)}
    return out


def spec_map(dataset):
    path = MAPPING_CSV.format(dataset=dataset)
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"{path} is empty")
    for field in ("sample_id", "original_id"):
        if field not in rows[0]:
            raise RuntimeError(f"{path} has no '{field}' column")
    return {r["sample_id"]: r["original_id"] for r in rows}


def spec_of(mapping, case, dataset):
    return mapping.get(str(case))


def collect_per_spec(column, dataset, model, mapping, carry_forward=False,
                     unmapped=None, runs=None):
    buckets = defaultdict(lambda: defaultdict(list))

    for run in (runs_for(column, model) if runs is None else runs):
        group_dir = group_dir_for(column, dataset, model, run)
        if not os.path.isdir(group_dir):
            continue
        base_dir = group_dir_for(BASELINE, dataset, model, run)
        for case in sorted(os.listdir(group_dir)):
            case_dir = os.path.join(group_dir, case)
            if not os.path.isdir(case_dir):
                continue

            vals, _carried = score_case(case_dir, base_dir, case, carry_forward)
            if vals is None or not vals["evaluated"]:
                continue

            spec = spec_of(mapping, case, dataset)
            if spec is None:
                if unmapped is not None:
                    unmapped.add((dataset, model, str(case)))
                continue

            for key in SCORE_KEYS:
                buckets[spec][key].append(vals[key])
            buckets[spec]["struct_or_logic"].append(
                float(vals["struct"] > 0 or vals["logic"] > 0))
            buckets[spec]["any_issue"].append(
                float(vals["struct"] + vals["logic"] + vals["neuro"] > 0))
            buckets[spec]["_n_samples"].append(1.0)

    out = {}
    for spec, series in buckets.items():
        row = {k: mean(v) for k, v in series.items() if k != "_n_samples"}
        row["_n_samples"] = len(series["_n_samples"])
        out[spec] = row
    return out


def aggregate_specs(per_spec, column, dataset, model):
    if not per_spec:
        return None
    specs = sorted(per_spec)
    row = {
        "dataset": dataset, "model": model, "column": column,
        "scope": "per_specification",
        "n_specifications": len(specs),
        "n_diagram_scores": sum(per_spec[s]["_n_samples"] for s in specs),
        "runs": ",".join(str(r) for r in runs_for(column, model)),
        "n_runs": len(runs_for(column, model)),
        "samples_per_spec_min": min(per_spec[s]["_n_samples"] for s in specs),
        "samples_per_spec_max": max(per_spec[s]["_n_samples"] for s in specs),
        "samples_per_spec_mean": mean(per_spec[s]["_n_samples"] for s in specs),
    }
    for key in SCORE_KEYS:
        row[f"avg_{key}"] = mean(per_spec[s][key] for s in specs)
    row["exp_specs_struct_or_logic"] = sum(
        per_spec[s]["struct_or_logic"] for s in specs)
    row["exp_specs_any_issue"] = sum(per_spec[s]["any_issue"] for s in specs)
    row["frac_specs_struct_or_logic"] = (row["exp_specs_struct_or_logic"]
                                         / len(specs))
    return row
