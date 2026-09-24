import os

import table_common as T
import abl_data as D
import sig_table

SIG = D.SIG
RC = D.RC

TOOLS = [("abscon", "AbsCon", "abscon"), ("ladex", "LADEX", "ladex")]


def main():
    summary = D.summary()
    written = []
    for kind, tname, slug in TOOLS:
        records, pairs = D.significance(kind)
        header, csv_rows = sig_table.build(
            records, pairs, SIG.SIG_ROW_DATASETS,
            SIG.SIG_METRIC_FAMILIES[kind],
            metric_label=lambda m: SIG.SIG_METRIC_LABEL[m],
            head_label=lambda c: D.A.COLUMN_LABEL[c],
                        model_label_fn=RC.model_label,
            summary=summary, sd_field=lambda m: "avg_" + m)
        header = ["scorer"] + header
        csv_rows = [[tname] + r for r in csv_rows]
        base = os.path.join(T.BASE_DIR,
                            "rq1_ablation_baseline_significance_" + slug)
        written.append(T.write_csv(base + ".csv", header, csv_rows))
    return written


if __name__ == "__main__":
    for p in main():
        print(f"Wrote {p}")
