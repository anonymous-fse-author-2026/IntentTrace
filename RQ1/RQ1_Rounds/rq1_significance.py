import os

import table_common as T
import rq1_data as D
import sig_table

SIG = D.SIG
RC = D.RC
BASE = os.path.join(T.BASE_DIR, "rq1_significance")


def main():
    records, pairs = D.significance("neuro")
    header, csv_rows = sig_table.build(
        records, pairs, SIG.SIG_ROW_DATASETS,
        SIG.SIG_METRIC_FAMILIES["neuro"],
        metric_label=lambda m: SIG.SIG_METRIC_LABEL[m],
        head_label=D.R.sig_head_label,
                model_label_fn=RC.model_label,
        summary=D.summary(), sd_field=lambda m: "avg_" + m)
    return [T.write_csv(BASE + ".csv", header, csv_rows)]


if __name__ == "__main__":
    for p in main():
        print(f"Wrote {p}")
