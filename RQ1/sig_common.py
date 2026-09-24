from statistics import mean

from scipy.stats import wilcoxon

SIG_METRICS = ["precision", "recall", "f1"]

SIG_METRIC_FAMILIES = {
    "neuro": SIG_METRICS,
    "abscon": [f"abscon_{m}" for m in SIG_METRICS],
    "ladex": [f"ladex_{m}" for m in SIG_METRICS],
}

SIG_METRIC_LABEL = {"precision": "Precision", "recall": "Recall", "f1": "F1"}
for _pre in ("abscon", "ladex"):
    for _m in SIG_METRICS:
        SIG_METRIC_LABEL[f"{_pre}_{_m}"] = SIG_METRIC_LABEL[_m]

SIG_ROW_DATASETS = [("Industry", ["gemini", "qwen"]),
                    ("PAGED", ["gemini", "qwen"])]

def compute_a12(a_vals, b_vals):
    n_a, n_b = len(a_vals), len(b_vals)
    if not n_a or not n_b:
        return float("nan")
    merged = sorted([(v, 0) for v in a_vals] + [(v, 1) for v in b_vals])
    ranks = {}
    i = 0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_a = sum(ranks[k] for k, (_v, g) in enumerate(merged) if g == 0)
    return (rank_sum_a / n_a - (n_a + 1) / 2.0) / n_b


def a12_magnitude(a12):
    if a12 != a12:
        return "n/a"
    if a12 >= 0.71 or a12 <= 0.29:
        return "large"
    if a12 >= 0.64 or a12 <= 0.36:
        return "medium"
    if a12 >= 0.56 or a12 <= 0.44:
        return "small"
    return "negligible"


def benjamini_hochberg(pvals):
    idx = [i for i, p in enumerate(pvals) if p == p]
    out = [float("nan")] * len(pvals)
    m = len(idx)
    if not m:
        return out
    order = sorted(idx, key=lambda i: pvals[i])
    running = 1.0
    for rank, i in zip(range(m, 0, -1), reversed(order)):
        running = min(running, m * pvals[i] / rank)
        out[i] = min(1.0, running)
    return out


def paired_test(a_vals, b_vals):
    n = len(a_vals)
    diffs = [x - y for x, y in zip(a_vals, b_vals)]
    out = {
        "n_A": n, "n_B": n,
        "n_pairs": n,
        "n_win": sum(1 for d in diffs if d > 0),
        "n_loss": sum(1 for d in diffs if d < 0),
        "n_tie": sum(1 for d in diffs if d == 0),
        "A12": float("nan"),
        "p_value": float("nan"), "test": "no common diagrams",
        "mean_A": float("nan"), "mean_B": float("nan"),
        "mean_diff": float("nan"),
    }
    if not n:
        out["A12_effect"] = a12_magnitude(float("nan"))
        return out

    out["mean_A"] = mean(a_vals)
    out["mean_B"] = mean(b_vals)
    out["mean_diff"] = mean(diffs)
    out["A12"] = compute_a12(a_vals, b_vals)

    if all(d == 0 for d in diffs):
        out["test"] = "all differences zero"
    else:
        try:
            _stat, p = wilcoxon(a_vals, b_vals)
            out["p_value"] = p
            out["test"] = "wilcoxon"
        except ValueError as exc:
            out["test"] = f"error: {exc}"

    out["A12_effect"] = a12_magnitude(out["A12"])
    return out


def run_block(dataset, model, comparisons, per_diagram, label_fn,
              model_label_fn=None, out_records=None, metrics=None):
    metrics = SIG_METRICS if metrics is None else list(metrics)
    usable = [(a, b) for a, b in comparisons
              if per_diagram.get(a) and per_diagram.get(b)]
    if not usable:
        return []

    block = []
    for col_a, col_b in usable:
        da, db = per_diagram[col_a], per_diagram[col_b]
        keys = sorted(set(da) & set(db))
        pair_runs = sorted({k[0] for k in keys},
                           key=lambda s: int(s) if str(s).isdigit() else s)
        for metric in metrics:
            rec = paired_test([da[k][metric] for k in keys],
                              [db[k][metric] for k in keys])
            rec.update({
                "dataset": dataset, "model": model,
                "runs": ",".join(str(x) for x in pair_runs),
                "n_runs": len(pair_runs),
                "group_A": col_a, "group_B": col_b,
                "label_A": label_fn(col_a), "label_B": label_fn(col_b),
                "metric": metric,
            })
            block.append(rec)

    for rec, p_adj in zip(block,
                          benjamini_hochberg([r["p_value"] for r in block])):
        rec["p_value_bh"] = p_adj
        rec["significant_bh_0.05"] = bool(p_adj == p_adj and p_adj < 0.05)

    if out_records is not None:
        out_records.extend(block)
    return block
