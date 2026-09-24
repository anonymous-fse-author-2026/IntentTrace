import table_common as T


def build(records, pairs, row_blocks, metrics, metric_label,
          head_label, model_label_fn, summary=None,
          sd_field=None):
    by_key = {(r["dataset"], r["model"], r["group_A"], r["group_B"],
               r["metric"]): r for r in records}

    header = ["dataset", "model", "comparison", "metric",
              "n_pairs", "mean_A", "sd_A", "mean_B", "sd_B", "mean_diff",
              "p_value", "p_value_bh", "significant_bh_0.05",
              "A12", "A12_effect"]
    csv_rows = []
    for dataset, models in row_blocks:
        for model in models:
            for col_a, col_b in pairs:
                for metric in metrics:
                    rec = by_key.get((dataset, model, col_a, col_b, metric))
                    if rec is None:
                        continue
                    csv_rows.append([
                        dataset, model_label_fn(model),
                        "%s vs %s" % (head_label(col_a), head_label(col_b)),
                        metric_label(metric), rec["n_pairs"],
                        _f(rec["mean_A"]),
                        _sd(summary, dataset, model, col_a, metric, sd_field),
                        _f(rec["mean_B"]),
                        _sd(summary, dataset, model, col_b, metric, sd_field),
                        _f(rec["mean_diff"]),
                        _g(rec["p_value"]), _g(rec["p_value_bh"]),
                        rec["significant_bh_0.05"],
                        _f(rec["A12"]), rec["A12_effect"]])
    return header, csv_rows


def _f(v):
    return "" if v is None or v == "" or v != v else "%.4f" % v


def _g(v):
    return "" if v is None or v == "" or v != v else "%.6g" % v


def _sd(summary, dataset, model, column, metric, sd_field):
    if not summary or not sd_field:
        return ""
    row = summary.get((dataset, model, column))
    if row is None:
        return ""
    sd = T.sd_of(row, sd_field(metric))
    return "" if sd is None else "%.4f" % sd
