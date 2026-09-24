import os

import table_common as T
import rq1_data as D

RC = D.RC
BASE = os.path.join(T.BASE_DIR, "rq1_baseline_quality")

TOOLS = [("abscon", "AbsCon"), ("ladex", "LADEX (B-Match)")]
METRICS = [("precision", "P"), ("recall", "R"), ("f1", "F1")]
BLOCKS = [(d, m) for d in D.DATASETS for m in D.MODELS]


def field(tool, metric):
    return f"avg_{tool}_{metric}"


def main():
    rows = D.summary()
    cols = D.columns()

    header = ["tool", "dataset", "model", "round", "n_runs", "runs"]
    for _m, mname in METRICS:
        header += [mname, mname + "_sd"]

    csv_rows = []
    for tool, tname in TOOLS:
        for dataset, model in BLOCKS:
            for column in cols:
                row = rows.get((dataset, model, column))
                if row is None:
                    continue
                out = [tname, dataset,
                       RC.model_label(model), D.R.column_label(column),
                       row["n_runs"], row["runs"]]
                for metric, _mn in METRICS:
                    f = field(tool, metric)
                    sd = T.sd_of(row, f)
                    out += ["%.4f" % row[f],
                            "" if sd is None else "%.4f" % sd]
                csv_rows.append(out)

    return [T.write_csv(BASE + ".csv", header, csv_rows)]


if __name__ == "__main__":
    for p in main():
        print(f"Wrote {p}")
