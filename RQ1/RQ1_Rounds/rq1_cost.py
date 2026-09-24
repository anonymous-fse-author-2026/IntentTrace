import os

import table_common as T
import rq1_data as D

RC = D.RC
R = D.R
BASE = os.path.join(T.BASE_DIR, "rq1_cost")

KINDS = ["input", "output", "reasoning", "total"]
STAGES = ["match", "trace", "refine"]
BLOCKS = [(d, m) for d in D.DATASETS for m in D.MODELS]


def per_call_diagram(row, stage, kind):
    n = row[f"{stage}_diagrams"]
    if not n:
        return 0.0
    return row[f"{stage}_{kind}_tokens"] / n


def main():
    rows = D.summary()
    cols = D.columns()

    header = ["dataset", "model", "round", "charged_stages", "n_runs", "runs",
              "n_cases"]
    for stage in STAGES:
        header.append(f"{stage}_diagrams")
        for kind in KINDS:
            header.append(f"{stage}_{kind}_per_diagram")
    for kind in KINDS:
        header.append(f"combined_{kind}_per_diagram")
    header.append("usd_per_diagram")

    csv_rows = []
    for dataset, model in BLOCKS:
        for column in cols:
            row = rows.get((dataset, model, column))
            if row is None:
                continue
            active = [s for s in STAGES if s in R.charged_buckets(column)
                      and row[f"{s}_diagrams"] > 0]
            out = [dataset, RC.model_label(model),
                   R.column_label(column), "+".join(active),
                   row["n_runs"], row["runs"], "%.0f" % row["n_cases"]]
            for stage in STAGES:
                out.append("%.1f" % row[f"{stage}_diagrams"])
                for kind in KINDS:
                    out.append("%.0f" % per_call_diagram(row, stage, kind))
            for kind in KINDS:
                out.append("%.0f" % sum(per_call_diagram(row, s, kind)
                                        for s in active))
            usd = sum(row[f"{s}_usd"] / row[f"{s}_diagrams"]
                      for s in active if row[f"{s}_diagrams"])
            out.append("" if not usd else "%.4f" % usd)
            csv_rows.append(out)

    written = [T.write_csv(BASE + ".csv", header, csv_rows)]
    written += T.emit_figure(BASE, lambda a: figure(rows, cols, a), 0.08)
    return written


def figure(rows, cols, labelled=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(len(D.DATASETS), len(D.MODELS), sharey=True,
                             squeeze=False,
                             figsize=(8.4 * len(D.MODELS),
                                      4.2 * len(D.DATASETS)))
    handles, labels = {}, []
    slot_w = 0.82 / len(STAGES)
    last = len(D.DATASETS) - 1

    for r, dataset in enumerate(D.DATASETS):
        for ci, model in enumerate(D.MODELS):
            ax = axes[r][ci]
            for xi, column in enumerate(cols):
                row = rows.get((dataset, model, column))
                if row is None:
                    continue
                active = [s for s in STAGES
                          if s in R.charged_buckets(column)
                          and row[f"{s}_diagrams"] > 0]
                for slot, stage in enumerate(active):
                    off = (slot - (len(active) - 1) / 2) * slot_w
                    stacked_bar(ax, xi + off, slot_w * 0.92,
                                R.STAGE_SHADES[stage], row, stage,
                                handles, labels, labelled,
                                lambda k, s=stage:
                                f"{R.BUCKET_LABEL[s]}: {k.capitalize()}")
            RC.finish_cost_axis(
                ax, f"{dataset} / {RC.model_label(model)}",
                np.arange(len(cols)), [R.column_label(c) for c in cols], "",
                {}, xlabel="Refinement" if r == last else None)
            ax.set_ylabel("Average Tokens per Diagram" if ci == 0 else "")

    if labelled:
        RC.cost_legend(fig, handles, labels, 0.04)
    fig.subplots_adjust(hspace=0.20)
    return fig


def stacked_bar(ax, x, width, shades, row, stage, handles, labels,
                labelled, label_fmt):
    bottom = 0.0
    for kind, color in shades:
        h = ax.bar(x, per_call_diagram(row, stage, kind), width,
                   bottom=bottom, color=color, edgecolor="white",
                   linewidth=0.6)
        bottom += per_call_diagram(row, stage, kind)
        key = label_fmt(kind)
        if key not in handles:
            handles[key] = h
            labels.append(key)
    if bottom > 0 and labelled:
        ax.annotate(f"{bottom/1000:.1f}k", (x, bottom),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=7.5 if width < 0.4 else 8, fontweight="bold")
    return bottom


if __name__ == "__main__":
    for p in main():
        print(f"Wrote {p}")
