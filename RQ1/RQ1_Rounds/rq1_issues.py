import os

import table_common as T
import rq1_data as D

RC = D.RC
BASE = os.path.join(T.BASE_DIR, "rq1_issues")

PANELS = RC.ISSUE_PANELS_MERGED
BLOCKS = [(d, m) for d in D.DATASETS for m in D.MODELS]


def main():
    rows = D.summary()
    cols = D.columns()

    header = ["dataset", "model", "round", "n_cases", "n_runs", "runs"]
    for _f, name in PANELS:
        header += [name, name + "_sd"]

    csv_rows = []
    for dataset, model in BLOCKS:
        for column in cols:
            row = rows.get((dataset, model, column))
            if row is None:
                continue
            out = [dataset, RC.model_label(model),
                   D.R.column_label(column), "%.0f" % row["n_cases"],
                   row["n_runs"], row["runs"]]
            for field, _name in PANELS:
                sd = T.sd_of(row, field)
                out += ["%.1f" % row[field],
                        "" if sd is None else "%.1f" % sd]
            csv_rows.append(out)

    written = [T.write_csv(BASE + ".csv", header, csv_rows)]
    written += T.emit_figure(BASE, lambda a: figure(rows, cols, a), 0.22)
    return written


def figure(rows, cols, labelled=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(cols), dtype=float)

    def vals(dataset, model, field):
        ys, es = [], []
        for c in cols:
            row = rows.get((dataset, model, c))
            ys.append(float("nan") if row is None else row[field])
            sd = None if row is None else T.sd_of(row, field)
            es.append(0.0 if sd is None else sd)
        return np.array(ys, dtype=float), np.array(es, dtype=float)

    top = 0.0
    for dataset in D.DATASETS:
        for model in D.MODELS:
            for field, _n in PANELS:
                ys, es = vals(dataset, model, field)
                if len(ys):
                    top = max(top, float(np.nanmax(ys + es)))
    ylim = (-top * 0.10 - 1, top * 1.22 + 1)

    fig, axes = plt.subplots(len(D.DATASETS), len(PANELS), squeeze=False,
                             figsize=(5.5 * len(PANELS),
                                      4.8 * len(D.DATASETS)))
    last = len(D.DATASETS) - 1
    for r, dataset in enumerate(D.DATASETS):
        for ax, (field, kind) in zip(axes[r], PANELS):
            pts = []
            for i, model in enumerate(D.MODELS):
                st = RC.style_for(model, i)
                ys, es = vals(dataset, model, field)
                ax.errorbar(x, ys, yerr=es, marker=st["marker"],
                            color=st["color"], lw=2, capsize=2.5,
                            elinewidth=0.9,
                            label=RC.model_label(model))
                pts.append((x, ys, es, st["color"]))
            ax.set_ylim(*ylim)
            ax.set_xlim(x[0] - 0.42, x[-1] + 0.42)
            if labelled:
                T.label_points(ax, pts, RC.fmt_count)
            ax.set_title(f"{dataset}: {kind}")
            ax.set_xticks(x)
            ax.set_xticklabels([D.R.column_label(c) for c in cols],
                               fontsize=8.5)
            if r == last:
                ax.set_xlabel("Refinement")
            ax.set_ylabel(RC.ISSUES_YLABEL)
            ax.tick_params(axis="y", labelleft=True)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.subplots_adjust(hspace=0.34)
    return fig


if __name__ == "__main__":
    for p in main():
        print(f"Wrote {p}")
