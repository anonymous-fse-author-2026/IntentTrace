#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from IntentTrace import prompts
from IntentTrace.constraint_checker import ConstraintChecking
from IntentTrace.evaluation import format_structural_logical_feedback
from IntentTrace.grammar import parse_process
from IntentTrace.llm import LLMCaller

BASE_ROOT = REPO / "Results" / "Round0" / "IntentTrace"
OUT_ROOT = REPO / "Results" / "Round0" / "Direct-LLM"
DATASET_ROOT = REPO / "Datasets" / "Sampled"
DATASETS = ["Industry", "PAGED"]
FAMILIES = ["gemini", "gemma", "qwen"]
LABEL = "llm_semantic_feedback"
NO_ISSUES = "no semantic issues found"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def dsl_from_grammar(grammar_path: Path) -> str:
    try:
        data = json.loads(grammar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    accepted = data.get("accepted") if isinstance(data, dict) else None
    if not isinstance(accepted, list):
        return ""
    return "\n".join(str(stmt) for stmt in accepted)


def family_from_model_name(name: str) -> str:
    base = name.split("-", 1)[0].lower()
    if base in FAMILIES:
        return base
    raise ValueError(f"Cannot infer model family from '{name}'; pass --family.")


def cost_from_call(call: Dict[str, Any]) -> Dict[str, Any]:
    reasoning = call.get("reasoning_tokens")
    return {
        "model_name": call.get("model_name"),
        "reasoning_effort": call.get("reasoning_effort"),
        "total_calls": 1,
        "total_errors": 1 if call.get("error") else 0,
        "total_prompt_tokens": call.get("prompt_tokens") or 0,
        "total_completion_tokens": call.get("completion_tokens") or 0,
        "total_reasoning_tokens": reasoning if isinstance(reasoning, (int, float)) else None,
        "total_tokens": call.get("total_tokens") or 0,
        "total_cost": call.get("cost"),
        "calls": [call],
    }


def combine_feedback(llm_critique: str, struct_logical: str) -> str:
    critique = (llm_critique or "").strip()
    if critique.lower().rstrip(".") == NO_ISSUES:
        critique = ""
    return "\n\n".join(p for p in (critique, (struct_logical or "").strip()) if p).strip()


def list_model_dirs(dataset: str, model_arg: Optional[str]) -> List[Path]:
    root = BASE_ROOT / dataset
    if not root.is_dir():
        return []
    folders = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not model_arg:
        return folders
    exact = [p for p in folders if p.name == model_arg]
    return exact or [p for p in folders if p.name.startswith(f"{model_arg}-")]


def list_sample_dirs(model_dir: Path) -> List[Path]:
    return [
        p for p in sorted(model_dir.iterdir(), key=lambda p: (len(p.name), p.name))
        if p.is_dir() and (p / f"{p.name}.grammar").is_file()
    ]


async def process_sample(dataset: str, model: str, sample_dir: Path,
                         llm: LLMCaller, overwrite: bool,
                         max_retries: int = 3) -> str:
    sid = sample_dir.name
    out_dir = OUT_ROOT / dataset / model / sid
    raw_path = out_dir / f"{sid}.feedback_llm_raw"

    if raw_path.exists() and raw_path.read_text(encoding="utf-8").strip() and not overwrite:
        return "skip"

    dsl = dsl_from_grammar(sample_dir / f"{sid}.grammar")
    if not dsl:
        return "no-dsl"
    description = _read(DATASET_ROOT / dataset / "GT" / "PD" / f"{sid}.txt")
    if not description.strip():
        return "no-description"

    report = parse_process(dsl)
    constraints = (
        ConstraintChecking(diagram_json=report.model).validate()
        if report.model.get("elements") else {}
    )
    struct_logical = format_structural_logical_feedback(constraints, report.model)
    prompt = prompts.build_direct_llm_feedback_prompt(description, dsl)

    last = "no response"
    for attempt in range(1, max_retries + 1):
        result = await llm.call_async([{"role": "user", "content": prompt}], label=LABEL)
        if result.error:
            last = result.error
        elif not (result.response or "").strip():
            last = "empty response"
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{sid}.feedback").write_text(
                combine_feedback(result.response, struct_logical), encoding="utf-8")
            raw_path.write_text(result.response, encoding="utf-8")
            (out_dir / f"{sid}.grammar").write_text(json.dumps({
                "ok": report.ok,
                "accepted_count": len(report.accepted),
                "rejected_count": len(report.rejected),
                "accepted": report.accepted,
                "rejected": report.rejected,
            }, indent=2), encoding="utf-8")
            (out_dir / f"{sid}.constraints").write_text(
                json.dumps(constraints, indent=2), encoding="utf-8")
            (out_dir / f"{sid}.cost").write_text(
                json.dumps(cost_from_call(result.to_dict()), indent=2), encoding="utf-8")
            return "ok"
        if attempt < max_retries:
            await asyncio.sleep(2 ** (attempt - 1))
    return f"error: {last}"


async def run_model(dataset: str, model_dir: Path, family: Optional[str],
                    reasoning_effort: Optional[str], diag: Optional[str],
                    limit: Optional[int], overwrite: bool) -> Dict[str, int]:
    llm = LLMCaller(model=family or family_from_model_name(model_dir.name),
                    reasoning_effort=reasoning_effort)
    samples = list_sample_dirs(model_dir)
    if diag:
        samples = [p for p in samples if p.name == diag]
    if limit is not None:
        samples = samples[:limit]

    print(f"=== {dataset}/{model_dir.name} (model='{llm.model_name}') "
          f"{len(samples)} diagrams ===", flush=True)
    counts: Dict[str, int] = {}
    for idx, sample_dir in enumerate(samples, start=1):
        status = await process_sample(dataset, model_dir.name, sample_dir, llm, overwrite)
        key = "error" if status.startswith("error") else status
        counts[key] = counts.get(key, 0) + 1
        print(f"[{idx}/{len(samples)}] {dataset}/{model_dir.name} "
              f"{sample_dir.name}: {status}", flush=True)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate the Direct-LLM feedback baseline into "
                    "Results/Round0/Direct-LLM.")
    ap.add_argument("--dataset", choices=DATASETS, help="dataset (default: all).")
    ap.add_argument("--model", help="base run folders to process, and the feedback LLM "
                                    "to use. A family name ('gemini') processes that "
                                    "family's folders with that LLM; a folder name "
                                    "('gemini-3') processes just that one. "
                                    "Default: all folders, each with its own family.")
    ap.add_argument("--family", choices=FAMILIES,
                    help="force the feedback LLM regardless of folder (default: "
                         "inferred from --model or the folder name).")
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--diag", help="only this diagram id.")
    ap.add_argument("--limit", type=int, help="cap diagrams per model folder.")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate feedback even when it already exists.")
    args = ap.parse_args()

    family = args.family
    if family is None and args.model in FAMILIES:
        family = args.model

    grand: Dict[str, int] = {}
    start = time.perf_counter()
    for dataset in ([args.dataset] if args.dataset else DATASETS):
        model_dirs = list_model_dirs(dataset, args.model)
        if not model_dirs:
            suffix = f" matching --model '{args.model}'" if args.model else ""
            print(f"[{dataset}] no base model folders{suffix}")
            continue
        for model_dir in model_dirs:
            counts = asyncio.run(run_model(
                dataset, model_dir, family, args.reasoning_effort,
                args.diag, args.limit, args.overwrite,
            ))
            for key, value in counts.items():
                grand[key] = grand.get(key, 0) + value
    print(f"=== TOTAL: {grand} in {time.perf_counter() - start:.1f}s ===")
    return 1 if grand.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
