import csv
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RQ1_DIR = os.path.dirname(BASE_DIR)
if RQ1_DIR not in sys.path:
    sys.path.insert(0, RQ1_DIR)


def has_spread(row):
    try:
        return int(float(row.get("n_runs", 0) or 0)) > 1
    except (TypeError, ValueError):
        return False


def sd_of(row, field):
    if not has_spread(row):
        return None
    val = row.get(field + "_sd")
    return None if val is None else float(val)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def emit_figure(base, draw, wspace):
    import report_common as RC

    out = []
    for suffix, labelled in ((".pdf", True), (".svg", False)):
        fig = draw(labelled)
        if not labelled:
            strip_text(fig)
        out.append(RC.save_fig(fig, base + suffix, wspace))
    return out


def label_points(ax, pts, fmt):
    for i, (xs, ys, sds, colour) in enumerate(pts):
        above = i % 2 == 0
        for x, y, sd in zip(xs, ys, sds):
            if y != y:
                continue
            ax.annotate(fmt(y) if callable(fmt) else fmt.format(y),
                        (x, y), textcoords="offset points",
                        xytext=(0, 7 if above else -13),
                        ha="center", fontsize=7.5, fontweight="bold",
                        color=colour)


def strip_text(fig):
    for ax in fig.axes:
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(labelbottom=False, labelleft=False,
                       labelright=False, labeltop=False)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        for t in list(ax.texts):
            t.remove()
    for t in list(fig.texts):
        t.remove()
    for lg in list(fig.legends):
        lg.remove()
