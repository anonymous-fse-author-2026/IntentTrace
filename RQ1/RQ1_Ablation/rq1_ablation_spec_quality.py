import os

import table_common as T
import abl_data as D

RC = D.RC
BASE = os.path.join(T.BASE_DIR, "rq1_ablation_spec_quality")

FIELDS = ["avg_precision", "avg_recall", "avg_f1"]
NAMES = ["P", "R", "F1"]
SL = "exp_specs_struct_or_logic"


def main():
    rows = D.spec_summary()

    header = ["dataset", "model", "approach", "n_specifications",
              "n_diagram_scores", "n_runs", "runs",
              "exp_specs_struct_or_logic", "exp_specs_struct_or_logic_sd"]
    for n in NAMES:
        header += [n, n + "_sd"]

    csv_rows = []
    for dataset, model in D.BLOCKS:
        for column in D.COLUMNS:
            row = rows.get((dataset, model, column))
            if row is None:
                continue
            sl = row.get(SL + "_sd")
            out = [dataset, RC.model_label(model),
                   D.A.COLUMN_LABEL[column],
                   "%.0f" % row["n_specifications"],
                   "%.0f" % row["n_diagram_scores"],
                   row["n_runs"], row["runs"],
                   "%.2f" % row[SL],
                   "" if sl is None else "%.2f" % sl]
            for f in FIELDS:
                sd = row.get(f + "_sd")
                out += ["%.4f" % row[f],
                        "" if sd is None else "%.4f" % sd]
            csv_rows.append(out)

    return [T.write_csv(BASE + ".csv", header, csv_rows)]


if __name__ == "__main__":
    for p in main():
        print(f"Wrote {p}")
