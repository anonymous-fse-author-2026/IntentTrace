import os

import table_common as T
import abl_data as D

RC = D.RC
BASE = os.path.join(T.BASE_DIR, "rq1_ablation_cost")

STAGES = D.A.STAGE_LABELS
COLUMNS = D.A.REFINE_TABLE_COLUMNS


def main():
    blocks = D.stage_costs()

    header = ["dataset", "model", "approach", "stage",
              "diagrams", "calls", "runs",
              "input", "output", "reasoning", "total", "usd"]
    csv_rows = []
    for dataset, model in D.BLOCKS:
        stats = blocks.get((dataset, model)) or {}
        for column in COLUMNS:
            per_stage = stats.get(column) or {}
            for stage, slabel in STAGES:
                row = per_stage.get(stage)
                if row is None:
                    continue
                out = [dataset, RC.model_label(model),
                       D.A.COLUMN_LABEL[column], slabel,
                       _n(row.get("n"), "{:.0f}"),
                       _n(row.get("calls"), "{:.0f}"),
                       row.get("runs", ""),
                       _n(row.get("input"), "{:.0f}"),
                       _n(row.get("output"), "{:.0f}"),
                       _n(row.get("reasoning"), "{:.0f}"),
                       _n(row.get("total"), "{:.0f}"),
                       _n(row.get("usd"), "{:.4f}")]
                csv_rows.append(out)

    return [T.write_csv(BASE + ".csv", header, csv_rows)]


def _n(val, fmt):
    return "" if val is None else fmt.format(val)


if __name__ == "__main__":
    for p in main():
        print(f"Wrote {p}")
