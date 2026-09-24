import os
import sys
from statistics import mean

RQ1_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(RQ1_DIR)
if RQ1_DIR not in sys.path:
    sys.path.insert(0, RQ1_DIR)

import report_common as RC
import sig_common as SIG

RESULTS_ROOT = os.path.join(REPO_ROOT, "Results")
APPROACH = "IntentTrace"
STRATEGY = "just_feedback"

MODELS = ["gemini", "qwen"]
DATASETS = ["Industry", "PAGED"]
RUNS = [1, 2, 3, 4, 5]
BASELINE = "baseline"

SINGLE_RUN_MODELS = {"gemini": 1}

COST_ALL_RUNS_COLUMNS = {"round1"}

COST_BUCKETS = ("match", "trace", "refine")

BUCKET_LABEL = {
    "match": "Element-Level Diagnostic",
    "trace": "Specification Trace Extraction",
    "refine": "Refinement",
}

STAGE_SHADES = {
    "match": [("input", "#1f4e79"), ("output", "#2e75b6"),
              ("reasoning", "#9dc3e6")],
    "trace": [("input", "#1b5e20"), ("output", "#43a047"),
              ("reasoning", "#a5d6a7")],
    "refine": [("input", "#993404"), ("output", "#e0700c"),
               ("reasoning", "#fdc086")],
}


def round_dir(n):
    return os.path.join(RESULTS_ROOT, f"Round{n}", APPROACH)


def round_col(n):
    return f"round{n}"


def col_round_no(column):
    return int(column[len("round"):])


def is_round(column):
    return column.startswith("round")


def stage_col(n):
    return BASELINE if n <= 0 else round_col(n)


def max_round():
    n = 1
    while os.path.isdir(round_dir(n + 1)):
        n += 1
    return n


def group_dir_for(column, dataset, model, run):
    n = 0 if column == BASELINE else col_round_no(column)
    return os.path.join(round_dir(n), dataset, f"{model}-{run}")


def ordinal(n):
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def column_label(column):
    if column == BASELINE:
        return "No Refinement"
    return f"{ordinal(col_round_no(column))} Round"


def sig_head_label(column):
    return column_label(column)


def charged_buckets(column):
    if column == BASELINE:
        return ("match", "trace", "refine")
    return ("match", "refine")


def has_feedback_errors(path):
    s, l, n = RC.count_feedback_tags(path)
    if not os.path.exists(path):
        return None
    return (s + l + n) > 0


def resolve_metrics(column, dataset, model, run, case, case_dir):
    data = RC.load_json(os.path.join(case_dir, f"{case}.metrics"))
    if data is None:
        return None, None, False
    feedback_path = os.path.join(case_dir, f"{case}.feedback")

    if is_round(column):
        prev_no = col_round_no(column) - 1
        prev_fb = os.path.join(
            group_dir_for(stage_col(prev_no), dataset, model, run), case,
            f"{case}.feedback")
        if has_feedback_errors(prev_fb) is False:
            for back in range(prev_no, -1, -1):
                prev_dir = os.path.join(
                    group_dir_for(stage_col(back), dataset, model, run), case)
                prev = RC.load_json(os.path.join(prev_dir, f"{case}.metrics"))
                if prev is not None and "precision_recall" in prev:
                    return prev, os.path.join(prev_dir, f"{case}.feedback"), True
    return data, feedback_path, False


def bucket_for(label):
    if label.startswith("refine"):
        return "refine"
    if label == "trace_builder":
        return "trace"
    if label == "root_cause_analysis":
        return None
    return "match"


def split_cost(case_dir, case, column):
    keep = charged_buckets(column)
    buckets = RC.empty_cost(COST_BUCKETS)
    for suffix in ("cost", "cost_refinement"):
        data = RC.load_json(os.path.join(case_dir, f"{case}.{suffix}"))
        if data is None:
            continue
        for call in data.get("calls") or []:
            bucket = bucket_for(str(call.get("label") or ""))
            if bucket in keep:
                RC.add_call(buckets[bucket], call)
    return buckets


def harmonic(precision, recall):
    total = precision + recall
    return (2 * precision * recall / total) if total else 0.0


def collect_run(column, dataset, model, run):
    group_dir = group_dir_for(column, dataset, model, run)
    if not os.path.isdir(group_dir):
        return None

    n_cases = carried = n_evaluated = 0
    struct_total = logic_total = neuro_total = 0
    struct_diag = logic_diag = neuro_diag = sl_union = 0
    with_issue = without_issue = 0
    abscon_perfect = ladex_perfect = 0
    precisions, recalls, f1s = [], [], []
    ab_prec, ab_rec, la_prec, la_rec = [], [], [], []
    ab_f1s, la_f1s = [], []
    cost = RC.empty_cost(COST_BUCKETS, diagrams=True)

    for case in sorted(os.listdir(group_dir)):
        case_dir = os.path.join(group_dir, case)
        if not os.path.isdir(case_dir):
            continue
        data, feedback_path, was_carried = resolve_metrics(
            column, dataset, model, run, case, case_dir)
        if data is None:
            continue

        n_cases += 1
        carried += int(was_carried)
        if not was_carried and "precision_recall" in data:
            n_evaluated += 1

        s_err, l_err, n_err = RC.count_feedback_tags(feedback_path)
        struct_total += s_err
        logic_total += l_err
        neuro_total += n_err
        struct_diag += s_err > 0
        logic_diag += l_err > 0
        neuro_diag += n_err > 0
        sl_union += (s_err > 0 or l_err > 0)
        if s_err + l_err + n_err > 0:
            with_issue += 1
        else:
            without_issue += 1

        pr = data.get("precision_recall") or {}
        precisions.append(RC.pr_precision(pr))
        recalls.append(RC.pr_recall(pr))
        f1s.append(RC.pr_f1(pr))

        ab_p, ab_r = RC.baseline_pair(data, "abscon")
        la_p, la_r = RC.baseline_pair(data, "ladex")
        ab_prec.append(ab_p)
        ab_rec.append(ab_r)
        la_prec.append(la_p)
        la_rec.append(la_r)
        ab_f1s.append(harmonic(ab_p, ab_r))
        la_f1s.append(harmonic(la_p, la_r))
        abscon_perfect += ab_p == 1.0 and ab_r == 1.0
        ladex_perfect += la_p == 1.0 and la_r == 1.0

        RC.add_run_cost(cost, split_cost(case_dir, case, column))

    if n_cases == 0:
        return None

    def avg(xs):
        return mean(xs) if xs else 0.0

    return {
        "column": column, "strategy": STRATEGY, "dataset": dataset,
        "model": model, "run": run,
        "n_cases": n_cases,
        "n_evaluated": n_evaluated,
        "carried_forward": carried,
        "n_diagrams_with_issue": with_issue,
        "n_diagrams_without_issue": without_issue,
        "n_abscon_perfect": abscon_perfect,
        "n_ladex_perfect": ladex_perfect,
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
        "avg_precision": avg(precisions),
        "avg_recall": avg(recalls),
        "avg_f1": avg(f1s),
        "avg_abscon_precision": avg(ab_prec),
        "avg_abscon_recall": avg(ab_rec),
        "avg_abscon_f1": avg(ab_f1s),
        "avg_ladex_precision": avg(la_prec),
        "avg_ladex_recall": avg(la_rec),
        "avg_ladex_f1": avg(la_f1s),
        "_cost": cost,
    }


AVG_FIELDS = [
    "n_cases", "n_evaluated", "carried_forward",
    "n_diagrams_with_issue", "n_diagrams_without_issue",
    "n_abscon_perfect", "n_ladex_perfect",
    "total_structural_errors", "avg_structural_errors",
    "n_diagrams_structural_issue",
    "total_logical_errors", "avg_logical_errors", "n_diagrams_logical_issue",
    "total_neuro_errors", "avg_neuro_errors", "n_diagrams_neuro_issue",
    "n_diagrams_struct_or_logic",
    "avg_precision", "avg_recall", "avg_f1",
    "avg_abscon_precision", "avg_abscon_recall", "avg_abscon_f1",
    "avg_ladex_precision", "avg_ladex_recall", "avg_ladex_f1",
]


def aggregate_runs(score_rows, cost_rows):
    out = RC.aggregate_runs(score_rows, cost_rows, AVG_FIELDS, COST_BUCKETS,
                            ("dataset", "model", "strategy", "column"))
    charged = charged_buckets(out["column"])
    out["charged_stages"] = "+".join(charged)
    return out


def collect_per_diagram(column, dataset, model, runs):
    values = {}
    for run in runs:
        group_dir = group_dir_for(column, dataset, model, run)
        if not os.path.isdir(group_dir):
            continue
        for case in sorted(os.listdir(group_dir)):
            case_dir = os.path.join(group_dir, case)
            if not os.path.isdir(case_dir):
                continue
            data, _, _ = resolve_metrics(column, dataset, model, run, case,
                                         case_dir)
            if data is None:
                continue
            pr = data.get("precision_recall")
            if not pr:
                continue
            ab_p, ab_r = RC.baseline_pair(data, "abscon")
            la_p, la_r = RC.baseline_pair(data, "ladex")
            values[(str(run), case)] = {
                "precision": RC.pr_precision(pr),
                "recall": RC.pr_recall(pr),
                "f1": RC.pr_f1(pr),
                "abscon_precision": ab_p,
                "abscon_recall": ab_r,
                "abscon_f1": harmonic(ab_p, ab_r),
                "ladex_precision": la_p,
                "ladex_recall": la_r,
                "ladex_f1": harmonic(la_p, la_r),
            }
    return values


def significance_pairs(columns):
    present = set(columns)
    rounds = sorted(col_round_no(c) for c in columns if is_round(c))

    consecutive = []
    for n in rounds:
        prev = BASELINE if n == 1 else round_col(n - 1)
        consecutive.append((round_col(n), prev))

    extra = [(round_col(n), BASELINE) for n in rounds if n >= 2]
    extra += [(round_col(n), round_col(1)) for n in rounds if n >= 2]

    families = []
    for group in (consecutive, extra):
        out = []
        for pair in group:
            if (pair not in out and pair[0] in present
                    and pair[1] in present):
                out.append(pair)
        families.append(out)
    families[1] = [p for p in families[1] if p not in families[0]]
    return families
