import os

import table_common as T
import abl_data as D

RC = D.RC
BASE = os.path.join(T.BASE_DIR, "rq1_ablation_quality")

FIELDS = list(D.A.QUALITY_TABLES["neuro"]["fields"])
NAMES = ["P", "R", "F1"]


def main():
    rows = D.summary()

    header = ["dataset", "model", "approach", "n_runs", "runs",
              "n_struct_or_logic", "n_struct_or_logic_sd"]
    for n in NAMES:
        header += [n, n + "_sd"]

    csv_rows = []
    for dataset, model in D.BLOCKS:
        for column in D.COLUMNS:
            row = rows.get((dataset, model, column))
            if row is None:
                continue
            sl = T.sd_of(row, "n_diagrams_struct_or_logic")
            out = [dataset, RC.model_label(model),
                   D.A.COLUMN_LABEL[column], row["n_runs"], row["runs"],
                   "%.1f" % row["n_diagrams_struct_or_logic"],
                   "" if sl is None else "%.1f" % sl]
            for f in FIELDS:
                sd = T.sd_of(row, f)
                out += ["%.4f" % row[f], "" if sd is None else "%.4f" % sd]
            csv_rows.append(out)

    return [T.write_csv(BASE + ".csv", header, csv_rows)]


if __name__ == "__main__":
    for p in main():
        print(f"Wrote {p}")
