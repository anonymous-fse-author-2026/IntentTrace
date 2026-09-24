import json
import re
from statistics import mean, pstdev

MODEL_DISPLAY = {"gemini": "Gemini", "qwen": "Qwen"}

MODEL_STYLE = {
    "gemini": dict(color="tab:blue", marker="o"),
    "qwen": dict(color="tab:red", marker="s"),
}
FALLBACK_STYLES = [
    dict(color="tab:green", marker="^"),
    dict(color="tab:purple", marker="D"),
]

TOKEN_KINDS = ["input", "output", "reasoning", "total"]

ISSUE_PANELS_MERGED = [
    ("n_diagrams_struct_or_logic", "Structural or Logical Diagnostics"),
    ("n_diagrams_neuro_issue", "Specification-Alignment Diagnostics"),
]

ISSUES_YLABEL = "# Diagrams with Non-Empty Diagnostics"

S_TAG = re.compile(r"\[S\d+\]")
L_TAG = re.compile(r"\[L\d+\]")
N_TAG = re.compile(r"\[N\d+\]")


def model_label(model):
    return MODEL_DISPLAY.get(model, model.capitalize())


def style_for(model, i):
    return MODEL_STYLE.get(model, FALLBACK_STYLES[i % len(FALLBACK_STYLES)])


def load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def count_feedback_tags(path):
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return 0, 0, 0
    return (len(S_TAG.findall(text)), len(L_TAG.findall(text)),
            len(N_TAG.findall(text)))


def pr_precision(pr):
    if not pr:
        return 0.0
    v = pr.get("precision")
    if v is None:
        v = pr.get("correctness_precision")
    return v or 0.0


def pr_recall(pr):
    if not pr:
        return 0.0
    v = pr.get("recall")
    if v is None:
        v = pr.get("completeness_recall")
    return v or 0.0


def pr_f1(pr):
    return (pr or {}).get("f1") or 0.0


def is_zeroed(data):
    data = data or {}
    sc = data.get("structural_conformance") or {}
    lc = data.get("logical_sanity_conformance") or {}
    s_ratio, l_ratio = sc.get("ratio"), lc.get("ratio")
    if s_ratio is None and l_ratio is None:
        pr = data.get("precision_recall") or {}
        return bool(pr.get("zeroed_for_constraint_violation"))
    return (s_ratio is not None and s_ratio < 1.0) or \
           (l_ratio is not None and l_ratio < 1.0)


def baseline_pair(data, tool):
    if is_zeroed(data):
        return 0.0, 0.0
    d = ((data or {}).get("baseline_scores") or {}).get(tool) or {}
    return d.get("precision") or 0.0, d.get("recall") or 0.0


def add_call(bucket, call):
    prompt = call.get("prompt_tokens") or 0
    completion = call.get("completion_tokens") or 0
    reasoning = call.get("reasoning_tokens") or 0
    bucket["input"] += prompt
    bucket["output"] += max(completion - reasoning, 0)
    bucket["reasoning"] += reasoning
    bucket["total"] += call.get("total_tokens") or (prompt + completion)
    bucket["cost"] += call.get("cost") or 0.0
    bucket["calls"] += 1


def empty_cost(buckets, diagrams=False):
    base = dict(input=0, output=0, reasoning=0, total=0, cost=0.0, calls=0)
    if diagrams:
        base["diagrams"] = 0
    return {b: dict(base) for b in buckets}


def add_run_cost(total, per_case):
    for bucket, vals in per_case.items():
        if vals["calls"] == 0:
            continue
        tgt = total[bucket]
        for k in ("input", "output", "reasoning", "total", "cost", "calls"):
            tgt[k] += vals[k]
        tgt["diagrams"] += 1


def aggregate_runs(score_rows, cost_rows, avg_fields, buckets, keys,
                   with_usd=True):
    out = {k: score_rows[0][k] for k in keys}
    out["n_runs"] = len(score_rows)
    out["runs"] = ",".join(str(r["run"]) for r in score_rows)

    for f in avg_fields:
        vals = [r[f] for r in score_rows]
        out[f] = mean(vals)
        out[f + "_sd"] = pstdev(vals) if len(vals) > 1 else 0.0

    for bucket in buckets:
        rows = [r for r in cost_rows if r["_cost"][bucket]["calls"] > 0]
        out[f"{bucket}_runs"] = ",".join(str(r["run"]) for r in rows) or "-"
        if with_usd:
            out[f"{bucket}_n_runs"] = len(rows)
        denom = mean([r["n_cases"] for r in rows]) if rows else 0.0
        for kind in TOKEN_KINDS:
            tot = mean([r["_cost"][bucket][kind] for r in rows]) if rows else 0.0
            out[f"{bucket}_{kind}_tokens"] = tot
            out[f"{bucket}_{kind}_per_diagram"] = tot / denom if denom else 0.0
        pairs = [("calls", "calls"), ("diagrams", "diagrams")]
        if with_usd:
            pairs.insert(0, ("usd", "cost"))
        for name, key in pairs:
            out[f"{bucket}_{name}"] = (
                mean([r["_cost"][bucket][key] for r in rows]) if rows else 0.0)
    return out


def fmt_count(v):
    return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"


def save_fig(fig, path, wspace):
    import matplotlib.pyplot as plt
    fig.subplots_adjust(wspace=wspace)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def thousands_axis(ax, axis="y"):
    from matplotlib.ticker import FuncFormatter

    def fmt(v, _pos):
        if v == 0:
            return "0"
        if abs(v) < 1000:
            return f"{v:,.0f}"
        k = v / 1000.0
        return f"{k:,.0f}k" if abs(k - round(k)) < 0.05 else f"{k:,.1f}k"

    getattr(ax, f"{axis}axis").set_major_formatter(FuncFormatter(fmt))


def finish_cost_axis(ax, title, xticks, ticks, ylabel, tick_kw, xlabel=None):
    ax.set_xticks(xticks)
    ax.set_xticklabels(ticks, **tick_kw)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="y", labelleft=True)
    thousands_axis(ax)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    ax.set_axisbelow(True)


def cost_legend(fig, handles, labels, y):
    fig.legend([handles[k] for k in labels], labels, fontsize=8.5, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, y), frameon=False,
               title_fontsize=8.5)
