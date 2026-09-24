import os

import table_common as T
import rq1_data as D

RC = D.RC
BASE = os.path.join(T.BASE_DIR, "rq1_quality")

FIELDS = [("avg_precision", "P"), ("avg_recall", "R"), ("avg_f1", "F1")]
BLOCKS = [(d, m) for d in D.DATASETS for m in D.MODELS]


def main():
    rows = D.summary()
    cols = D.columns()

    header = ["dataset", "model", "round", "n_runs", "runs"]
    for _f, name in FIELDS:
        header += [name, name + "_sd"]
    csv_rows = []
    for dataset, model in BLOCKS:
        for column in cols:
            row = rows.get((dataset, model, column))
            if row is None:
                continue
            out = [dataset, RC.model_label(model),
                   R_label(column), row["n_runs"], row["runs"]]
            for field, _name in FIELDS:
                sd = T.sd_of(row, field)
                out += ["%.4f" % row[field],
                        "" if sd is None else "%.4f" % sd]
            csv_rows.append(out)

    written = [T.write_csv(BASE + ".csv", header, csv_rows)]
    written += T.emit_figure(BASE, lambda a: figure(rows, cols, a), 0.12)
    return written


def R_label(column):
    return D.R.column_label(column)


def figure(rows, cols, labelled=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, len(D.DATASETS), figsize=(11.0, 4.4),
                             sharey=True)
    x = np.arange(len(cols), dtype=float)

    for ax, dataset in zip(np.atleast_1d(axes), D.DATASETS):
        f1_pts = []
        for i, model in enumerate(D.MODELS):
            st = RC.style_for(model, i)
            for field, name in FIELDS:
                ys, es = [], []
                for c in cols:
                    row = rows.get((dataset, model, c))
                    ys.append(float("nan") if row is None else row[field])
                    sd = None if row is None else T.sd_of(row, field)
                    es.append(0.0 if sd is None else sd)
                ys = np.array(ys, dtype=float)
                es = np.array(es, dtype=float)

                is_f1 = name == "F1"
                ax.errorbar(
                    x, ys, yerr=es, lw=2.4 if is_f1 else 1.5,
                    ls={"P": (0, (5, 1.6)), "R": (0, (1.4, 1.6)),
                        "F1": "-"}[name],
                    marker=st["marker"], ms=6 if is_f1 else 4.2,
                    markerfacecolor=st["color"] if is_f1 else "white",
                    markeredgecolor=st["color"], markeredgewidth=1.2,
                    color=st["color"], alpha=1.0 if is_f1 else 0.8,
                    capsize=2.5, elinewidth=0.9,
                    label=f"{RC.model_label(model)} {name}")
                if is_f1:
                    f1_pts.append((x, ys, es, st["color"]))

        ax.set_ylim(-0.08, 1.10)
        ax.set_xlim(x[0] - 0.42, x[-1] + 0.42)
        if labelled:
            T.label_points(ax, f1_pts, "{:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels([R_label(c) for c in cols], fontsize=8.5)
        ax.set_title(dataset)
        ax.set_xlabel("Refinement")
        ax.set_ylabel("Average Accuracy")
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=len(D.MODELS), loc="lower right",
                  columnspacing=1.4)

    return fig


if __name__ == "__main__":
    for p in main():
        print(f"Wrote {p}")
