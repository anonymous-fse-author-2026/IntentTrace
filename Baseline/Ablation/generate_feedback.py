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
from IntentTrace.cli import _input_trace_elements
from IntentTrace.constraint_checker import ConstraintChecking
from IntentTrace.element_matching import match_elements
from IntentTrace.evaluation import (
    format_consolidated_feedback,
    format_structural_logical_feedback,
)
from IntentTrace.grammar import parse_process
from IntentTrace.llm import LLMCaller, MODELS, resolve_model
from IntentTrace.parsed_model import build_parsed_model

BASE_ROOT = REPO / "Results" / "Round0" / "IntentTrace"
RESULTS_ROOT = REPO / "Results"
DEFAULT_OUT_ROOT = REPO / "Results_Ablation"
DATASET_ROOT = REPO / "Datasets" / "Sampled"

DATASETS = ["Industry", "PAGED"]
FAMILIES = ["gemini", "qwen"]

MODEL_INTERNAL = "Model-Internal"
ELEMENT_ALIGNMENT = "Element-Alignment"
DIRECT_LLM = "Direct-LLM"
VARIANTS = [MODEL_INTERNAL, ELEMENT_ALIGNMENT, DIRECT_LLM]

ELEMENT_LABEL = "element_mapping"
LLM_LABEL = "llm_semantic_feedback"
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


def empty_cost() -> Dict[str, Any]:
    return {
        "model_name": None,
        "reasoning_effort": None,
        "total_calls": 0,
        "total_errors": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_reasoning_tokens": None,
        "total_tokens": 0,
        "total_cost": None,
        "calls": [],
    }


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


def grammar_report(report: Any) -> Dict[str, Any]:
    return {
        "ok": report.ok,
        "accepted_count": len(report.accepted),
        "rejected_count": len(report.rejected),
        "accepted": report.accepted,
        "rejected": report.rejected,
    }


def combine_llm_feedback(critique: str, struct_logical: str) -> str:
    text = (critique or "").strip()
    if text.lower().rstrip(".") == NO_ISSUES:
        text = ""
    return "\n\n".join(p for p in (text, (struct_logical or "").strip()) if p).strip()


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


def out_dir_for(out_root: Path, variant: str, dataset: str, model: str,
                sid: str) -> Path:
    return out_root / "Round0" / variant / dataset / model / sid


def already_done(out_dir: Path, sid: str) -> bool:
    return (out_dir / f"{sid}.feedback").exists()


async def run_model_internal(sid: str, out_dir: Path, dsl: str,
                             _description: str, _llm: Optional[LLMCaller],
                             _model_id: Optional[str], _effort: Optional[str]) -> str:
    report = parse_process(dsl)
    model = report.model
    constraints = (ConstraintChecking(diagram_json=model).validate()
                   if model.get("elements") else {})
    feedback = format_structural_logical_feedback(constraints, model)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{sid}.feedback").write_text(feedback, encoding="utf-8")
    (out_dir / f"{sid}.grammar").write_text(
        json.dumps(grammar_report(report), indent=2), encoding="utf-8")
    (out_dir / f"{sid}.constraints").write_text(
        json.dumps(constraints, indent=2), encoding="utf-8")
    (out_dir / f"{sid}.cost").write_text(
        json.dumps(empty_cost(), indent=2), encoding="utf-8")
    return "ok"


async def run_element_alignment(sid: str, out_dir: Path, dsl: str,
                                description: str, llm: LLMCaller,
                                model_id: str, effort: str) -> str:
    if not description.strip():
        return "no-description"

    report = parse_process(dsl)
    model = report.model
    if not model.get("elements"):
        return "no-elements"
    constraints = ConstraintChecking(diagram_json=model).validate()

    mapping = await match_elements(
        input_text=description, model=build_parsed_model(model), llm_caller=llm,
        model_id=model_id, reasoning_effort=effort,
    )

    calls = llm.get_cost_summary().get("calls") or []
    matching = [c for c in calls if c.get("label") == ELEMENT_LABEL]
    if not matching:
        return "error: no element_mapping call recorded"
    call = matching[-1]
    if call.get("error"):
        return f"error: {call['error']}"

    mapping_dict = mapping.to_dict()
    feedback = format_consolidated_feedback(
        mapping_result=mapping,
        evaluation_result={},
        var_actor_result=None,
        constraints=constraints,
        model=model,
        input_trace_elements=_input_trace_elements(mapping_dict, description),
    )

    cost = cost_from_call(call)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{sid}.feedback").write_text(feedback, encoding="utf-8")
    (out_dir / f"{sid}.grammar").write_text(
        json.dumps(grammar_report(report), indent=2), encoding="utf-8")
    (out_dir / f"{sid}.constraints").write_text(
        json.dumps(constraints, indent=2), encoding="utf-8")
    (out_dir / f"{sid}.mapping").write_text(
        json.dumps(mapping_dict, indent=2), encoding="utf-8")
    (out_dir / f"{sid}.cost").write_text(json.dumps(cost, indent=2), encoding="utf-8")
    return "ok"


async def run_direct_llm(sid: str, out_dir: Path, dsl: str,
                         description: str, llm: LLMCaller,
                         _model_id: str, _effort: str,
                         max_retries: int = 3) -> str:
    if not description.strip():
        return "no-description"

    report = parse_process(dsl)
    model = report.model
    constraints = (ConstraintChecking(diagram_json=model).validate()
                   if model.get("elements") else {})
    struct_logical = format_structural_logical_feedback(constraints, model)
    prompt = prompts.build_direct_llm_feedback_prompt(description, dsl)

    last = "no response"
    for attempt in range(1, max_retries + 1):
        result = await llm.call_async([{"role": "user", "content": prompt}],
                                     label=LLM_LABEL)
        if result.error:
            last = result.error
        elif not (result.response or "").strip():
            last = "empty response"
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{sid}.feedback").write_text(
                combine_llm_feedback(result.response, struct_logical),
                encoding="utf-8")
            (out_dir / f"{sid}.feedback_llm_raw").write_text(
                result.response, encoding="utf-8")
            (out_dir / f"{sid}.grammar").write_text(
                json.dumps(grammar_report(report), indent=2), encoding="utf-8")
            (out_dir / f"{sid}.constraints").write_text(
                json.dumps(constraints, indent=2), encoding="utf-8")
            (out_dir / f"{sid}.cost").write_text(
                json.dumps(cost_from_call(result.to_dict()), indent=2),
                encoding="utf-8")
            return "ok"
        if attempt < max_retries:
            await asyncio.sleep(2 ** (attempt - 1))
    return f"error: {last}"


VARIANT_RUNNERS = {
    MODEL_INTERNAL: run_model_internal,
    ELEMENT_ALIGNMENT: run_element_alignment,
    DIRECT_LLM: run_direct_llm,
}

NEEDS_LLM = {ELEMENT_ALIGNMENT, DIRECT_LLM}


async def process_sample(variant: str, dataset: str, model: str,
                         sample_dir: Path, out_root: Path,
                         llm: Optional[LLMCaller],
                         model_id: Optional[str], effort: Optional[str]) -> str:
    sid = sample_dir.name
    out_dir = out_dir_for(out_root, variant, dataset, model, sid)
    if already_done(out_dir, sid):
        return "skip"

    dsl = dsl_from_grammar(sample_dir / f"{sid}.grammar")
    if not dsl:
        return "no-dsl"
    description = _read(DATASET_ROOT / dataset / "GT" / "PD" / f"{sid}.txt")

    return await VARIANT_RUNNERS[variant](
        sid, out_dir, dsl, description, llm, model_id, effort)


async def run_model(variant: str, dataset: str, model_dir: Path, out_root: Path,
                    family: Optional[str], reasoning_effort: Optional[str],
                    diag: Optional[str], limit: Optional[int]) -> Dict[str, int]:
    llm = model_id = effort = None
    if variant in NEEDS_LLM:
        resolved = family or family_from_model_name(model_dir.name)
        llm = LLMCaller(model=resolved, reasoning_effort=reasoning_effort)
        model_id = MODELS[resolve_model(resolved)]["model"]
        effort = reasoning_effort or str(
            MODELS[resolve_model(model_id)].get("reasoning_effort") or "none")

    samples = list_sample_dirs(model_dir)
    if diag:
        samples = [p for p in samples if p.name == diag]
    if limit is not None:
        samples = samples[:limit]

    label = llm.model_name if llm is not None else "no-llm"
    print(f"=== [{variant}] {dataset}/{model_dir.name} (model='{label}') "
          f"{len(samples)} diagrams ===", flush=True)

    counts: Dict[str, int] = {}
    for idx, sample_dir in enumerate(samples, start=1):
        status = await process_sample(variant, dataset, model_dir.name,
                                      sample_dir, out_root, llm, model_id, effort)
        key = "error" if status.startswith("error") else status
        counts[key] = counts.get(key, 0) + 1
        print(f"[{idx}/{len(samples)}] {variant} {dataset}/{model_dir.name} "
              f"{sample_dir.name}: {status}", flush=True)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate ablation feedback from the Round0 IntentTrace models. "
                    "Model-Internal is structural+logical only; Element-Alignment "
                    "adds element matching; Direct-LLM adds an LLM critique. Writes "
                    "to Results_Ablation unless --out is given; existing outputs are "
                    "never overwritten.")
    ap.add_argument("--variant", choices=VARIANTS, action="append",
                    help="ablation to generate, named after its output directory "
                         "(repeatable, default: all).")
    ap.add_argument("--dataset", choices=DATASETS, help="dataset (default: all).")
    ap.add_argument("--model", help="run folder ('gemini-3') or family ('gemini').")
    ap.add_argument("--family", choices=FAMILIES,
                    help="force the feedback LLM regardless of folder name.")
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--diag", help="only this diagram id.")
    ap.add_argument("--limit", type=int, help="cap diagrams per model folder.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT,
                    help=f"output root (default: {DEFAULT_OUT_ROOT.name}). Pass "
                         f"'{RESULTS_ROOT.name}' to write alongside the published "
                         f"results.")
    args = ap.parse_args()

    out_root = args.out.resolve()
    variants = args.variant or VARIANTS
    family = args.family
    if family is None and args.model in FAMILIES:
        family = args.model

    print(f"[out] {out_root}")
    grand: Dict[str, int] = {}
    start = time.perf_counter()
    for variant in variants:
        for dataset in ([args.dataset] if args.dataset else DATASETS):
            model_dirs = list_model_dirs(dataset, args.model)
            if not model_dirs:
                suffix = f" matching --model '{args.model}'" if args.model else ""
                print(f"[{variant}/{dataset}] no base model folders{suffix}")
                continue
            for model_dir in model_dirs:
                counts = asyncio.run(run_model(
                    variant, dataset, model_dir, out_root, family,
                    args.reasoning_effort, args.diag, args.limit,
                ))
                for key, value in counts.items():
                    grand[key] = grand.get(key, 0) + value

    print(f"=== TOTAL: {grand} in {time.perf_counter() - start:.1f}s ===")
    return 1 if grand.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
