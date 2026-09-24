import os

import table_common as T
import abl_data as D

RC = D.RC
BASE = os.path.join(T.BASE_DIR, "rq1_ablation_baseline_quality")

TOOLS = [("abscon", "AbsCon"), ("ladex", "LADEX (B-Match)")]
NAMES = ["P", "R", "F1"]


def main():
    rows = D.summary()

    header = ["scorer", "dataset", "model", "approach", "n_runs", "runs"]
    for n in NAMES:
        header += [n, n + "_sd"]

    csv_rows = []
    for tool, tname in TOOLS:
        fields = D.A.QUALITY_TABLES[tool]["fields"]
        for dataset, model in D.BLOCKS:
            for column in D.COLUMNS:
                row = rows.get((dataset, model, column))
                if row is None:
                    continue
                out = [tname, dataset,
                       RC.model_label(model), D.A.COLUMN_LABEL[column],
                       row["n_runs"], row["runs"]]
                for f in fields:
                    sd = T.sd_of(row, f)
                    out += ["%.4f" % row[f],
                            "" if sd is None else "%.4f" % sd]
                csv_rows.append(out)

    return [T.write_csv(BASE + ".csv", header, csv_rows)]


if __name__ == "__main__":
    for p in main():
        print(f"Wrote {p}")
