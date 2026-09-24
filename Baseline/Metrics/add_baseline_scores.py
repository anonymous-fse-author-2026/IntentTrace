from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Render.dsl_to_csv import dsl_to_csv
from abscon_eval import abscon_scores
import bmatch

RESULTS_ROOT = REPO_ROOT / "Results"
SAMPLED_ROOT = REPO_ROOT / "Datasets" / "Sampled"

DATASETS = ("Industry", "PAGED")

LADEX_THRESHOLD = 0.0


def gt_csv_path(dataset: str, diagram_id: str) -> Path:
    return SAMPLED_ROOT / dataset / "GT" / "PM_CSV" / f"{diagram_id}.txt"


def input_dsl_path(dataset: str, diagram_id: str) -> Path:
    return SAMPLED_ROOT / dataset / "DSL" / f"{diagram_id}.txt"


def model_dsl_path(metrics_path: Path, round_name: str, dataset: str,
                   diagram_id: str) -> Path:
    if round_name == "Round0":
        return input_dsl_path(dataset, diagram_id)
    return metrics_path.parent / f"{diagram_id}.refined"


def is_zeroed(metrics: Dict) -> bool:
    metrics = metrics or {}
    sc = metrics.get("structural_conformance") or {}
    lc = metrics.get("logical_sanity_conformance") or {}
    s_ratio, l_ratio = sc.get("ratio"), lc.get("ratio")
    if s_ratio is None and l_ratio is None:
        pr = metrics.get("precision_recall") or {}
        return bool(pr.get("zeroed_for_constraint_violation"))
    return (s_ratio is not None and s_ratio < 1.0) or \
           (l_ratio is not None and l_ratio < 1.0)


def ladex_scores(gt_csv: str, gen_csv: str) -> Dict[str, float]:
    try:
        corr = bmatch.evaluate_threshold_at_end_one_to_many(gen_csv, gt_csv, LADEX_THRESHOLD)
        comp = bmatch.evaluate_threshold_at_end_one_to_many(gt_csv, gen_csv, LADEX_THRESHOLD)
        return {"precision": corr["coverage_1_to_2"],
                "recall": comp["coverage_1_to_2"]}
    except Exception as exc:
        print(f"    LADEX scoring error: {exc}")
        return {"precision": 0.0, "recall": 0.0}


def score_pair(gt_csv_text: str, gen_dsl_text: str,
               metric: str) -> Dict[str, Dict[str, float]]:
    gen_csv_text = dsl_to_csv(gen_dsl_text)

    scores = {"ladex": {"precision": 0.0, "recall": 0.0},
              "abscon": {"precision": 0.0, "recall": 0.0}}

    if metric in ("both", "abscon"):
        ab = abscon_scores(gt_csv_text, gen_csv_text)
        scores["abscon"] = {"precision": ab["precision"], "recall": ab["recall"]}
    if metric in ("both", "ladex"):
        scores["ladex"] = ladex_scores(gt_csv_text, gen_csv_text)

    return scores


def iter_metrics(round_filter: Optional[str], approach_filter: Optional[str],
                 dataset_filter: Optional[str], model_filter: Optional[str],
                 only: Optional[set]):
    for round_dir in sorted(p for p in RESULTS_ROOT.iterdir() if p.is_dir()):
        if round_filter and round_dir.name != round_filter:
            continue
        for approach_dir in sorted(p for p in round_dir.iterdir() if p.is_dir()):
            if approach_filter and approach_dir.name != approach_filter:
                continue
            for ds_dir in sorted(p for p in approach_dir.iterdir() if p.is_dir()):
                if ds_dir.name not in DATASETS:
                    continue
                if dataset_filter and ds_dir.name != dataset_filter:
                    continue
                for run_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
                    if not run_matches(run_dir.name, model_filter):
                        continue
                    for case_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
                        did = case_dir.name
                        if only is not None and did not in only:
                            continue
                        mp = case_dir / f"{did}.metrics"
                        if mp.is_file():
                            yield (mp, round_dir.name, approach_dir.name,
                                   ds_dir.name, run_dir.name, did)


def run_matches(run: str, model_filter: Optional[str]) -> bool:
    if model_filter is None:
        return True
    return run == model_filter or run.startswith(f"{model_filter}-")


def process(args: argparse.Namespace) -> Tuple[int, int, int]:
    only = None
    if args.only:
        only = {tok.strip() for tok in args.only.split(",") if tok.strip()}

    updated = skipped = failed = 0

    gt_cache: Dict[Tuple[str, str], Optional[str]] = {}
    round0_cache: Dict[Tuple[str, str], Optional[Dict]] = {}

    def read_gt(dataset: str, did: str) -> Optional[str]:
        key = (dataset, did)
        if key not in gt_cache:
            p = gt_csv_path(dataset, did)
            gt_cache[key] = p.read_text(encoding="utf-8") if p.is_file() else None
        return gt_cache[key]

    for mp, rnd, approach, dataset, run, did in iter_metrics(
            args.round, args.approach, args.dataset, args.model, only):
        rel = mp.relative_to(REPO_ROOT)

        try:
            metrics = json.loads(mp.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[fail] bad JSON {rel}: {exc}")
            failed += 1
            continue

        if "baseline_scores" in metrics and not args.overwrite:
            skipped += 1
            continue

        gt_csv = read_gt(dataset, did)
        if gt_csv is None:
            print(f"[fail] missing GT CSV: {gt_csv_path(dataset, did)}")
            failed += 1
            continue

        cache_key = (dataset, did)
        if rnd == "Round0" and cache_key in round0_cache:
            scores = round0_cache[cache_key]
            if scores is None:
                failed += 1
                continue
        else:
            dsl_path = model_dsl_path(mp, rnd, dataset, did)
            if not dsl_path.is_file():
                print(f"[fail] missing model DSL: {dsl_path.relative_to(REPO_ROOT)}")
                if rnd == "Round0":
                    round0_cache[cache_key] = None
                failed += 1
                continue
            try:
                scores = score_pair(gt_csv, dsl_path.read_text(encoding="utf-8"),
                                    args.metric)
            except Exception as exc:
                print(f"[fail] {rel}: {exc}")
                if rnd == "Round0":
                    round0_cache[cache_key] = None
                failed += 1
                continue
            if rnd == "Round0":
                round0_cache[cache_key] = scores

        zeroed = is_zeroed(metrics)
        if zeroed:
            block = {"ladex": {"precision": 0.0, "recall": 0.0},
                     "abscon": {"precision": 0.0, "recall": 0.0}}
        else:
            block = {"ladex": dict(scores["ladex"]),
                     "abscon": dict(scores["abscon"])}

        flag = " (ZEROED)" if zeroed else ""
        print(f"[ok] {rel}: "
              f"LADEX p={block['ladex']['precision']:.3f} r={block['ladex']['recall']:.3f} | "
              f"AbsCon p={block['abscon']['precision']:.3f} r={block['abscon']['recall']:.3f}{flag}")

        if not args.dry_run:
            metrics["baseline_scores"] = block
            mp.write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                          encoding="utf-8")
        updated += 1

    return updated, skipped, failed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", default=None,
                    help="Only this round (e.g. 'Round0'). Default: all.")
    ap.add_argument("--approach", default=None,
                    help="Only this approach folder (e.g. 'IntentTrace'). "
                         "Default: all.")
    ap.add_argument("--dataset", choices=DATASETS, default=None,
                    help="Only this dataset. Default: both.")
    ap.add_argument("--model", default=None,
                    help="Only runs matching this model. A family name (e.g. "
                         "'qwen') matches qwen-1..qwen-5; a full run name (e.g. "
                         "'qwen-1') matches only that run. Default: all.")
    ap.add_argument("--only", default=None,
                    help="Comma-separated diagram ids (e.g. '1,2,3').")
    ap.add_argument("--metric", choices=("both", "abscon", "ladex"),
                    default="both",
                    help="Which baseline(s) to compute. 'abscon' avoids loading "
                         "the embedding model. Default: both.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute even if baseline_scores is already present.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would happen without writing files.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    u, s, f = process(args)
    mode = " (dry run, nothing written)" if args.dry_run else ""
    print(f"\n=== TOTAL{mode}: updated={u} skipped={s} failed={f} ===")
    return 1 if f else 0


if __name__ == "__main__":
    raise SystemExit(main())
