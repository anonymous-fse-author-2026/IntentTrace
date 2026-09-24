from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

_CA_BUNDLE = REPO_ROOT / "corp-ca.pem"
if _CA_BUNDLE.is_file():
    os.environ.setdefault("SSL_CERT_FILE", str(_CA_BUNDLE))
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(_CA_BUNDLE))

from . import prompts
from .llm import LLMCaller
from .grammar import parse_process
from .constraint_checker import ConstraintChecking
from .evaluation import (
    _STC_DESCRIPTIONS,
    _LSC_DESCRIPTIONS,
    _build_id_to_dsl,
    _build_ref_to_dsl,
    _conformance_ratio,
    _format_check_section,
)

RESULTS = HERE / "Results"
OUT_ROOT = HERE / "Results_Refinement"
DATASETS = ["Industry", "PAGED"]
VARIANTS = ["just_feedback", "just_rootcause", "rootcause_feedback_combined"]

_ID_RE = re.compile(r"^\s*-\s*\[([NSL]\d+)\]\s*(.*)$")
_INCLUDES_RE = re.compile(r"^(\s*-\s*Includes:\s*)(.*)$", re.IGNORECASE)


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


def build_feedback_id_map(feedback_text: str) -> dict[str, str]:
    id_map: dict[str, str] = {}
    for line in feedback_text.splitlines():
        m = _ID_RE.match(line)
        if m:
            id_map[m.group(1)] = m.group(2).strip()
    return id_map


def render_traces(trace_path: Path) -> tuple[str, str]:
    try:
        data = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "None", "None"

    def _render(traces, label: str) -> str:
        if not traces:
            return "None"
        lines = []
        for idx, trace in enumerate(traces):
            steps = [str(s).strip() for s in (trace or []) if str(s).strip()]
            lines.append(f"{label} {idx}: {' -> '.join(steps) if steps else '(empty trace)'}")
        return "\n".join(lines)

    src = _render(data.get("source_text_traces"), "input")
    mdl = _render(data.get("model_traces"), "model")
    return src, mdl


def render_input_elements(mapping_path: Path, process_description: str) -> str:
    try:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "None"
    if not isinstance(mapping, dict):
        return "None"

    ordered: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    discovery = 0
    for category in ("matches", "mutations", "omissions"):
        for entry in mapping.get(category, []) or []:
            if not isinstance(entry, dict):
                continue
            for raw in entry.get("verbatim_texts", []) or []:
                text = str(raw).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                pos = process_description.find(text)
                if pos < 0:
                    pos = process_description.lower().find(text.lower())
                if pos < 0:
                    pos = sys.maxsize
                ordered.append((pos, discovery, text))
                discovery += 1

    if not ordered:
        return "None"
    ordered.sort(key=lambda item: (item[0], item[1]))
    return "\n".join(f"i{idx + 1}: {text}" for idx, (_, _, text) in enumerate(ordered))


def validate_refined_dsl(refined_dsl: str) -> dict:
    report = parse_process(refined_dsl)
    model = report.model
    grammar_report = {
        "ok": report.ok,
        "accepted_count": len(report.accepted),
        "rejected_count": len(report.rejected),
        "accepted": report.accepted,
        "rejected": report.rejected,
    }

    if not model.get("elements"):
        return {
            "grammar": grammar_report,
            "constraints": {},
            "feedback": "",
            "metrics": {
                "structural_conformance": {"passed": 0, "total": 0, "ratio": 1.0},
                "logical_sanity_conformance": {"passed": 0, "total": 0, "ratio": 1.0},
            },
        }

    constraints = ConstraintChecking(diagram_json=model).validate()

    ref_to_dsl = _build_ref_to_dsl(model)
    id_to_dsl = _build_id_to_dsl(ref_to_dsl)
    structural = _format_check_section(
        title="Structural Check", constraints=constraints, prefix="STC",
        descriptions=_STC_DESCRIPTIONS, id_prefix="S",
        id_to_dsl=id_to_dsl, ref_to_dsl=ref_to_dsl,
    )
    logical = _format_check_section(
        title="Logical Sanity Check", constraints=constraints, prefix="LSC",
        descriptions=_LSC_DESCRIPTIONS, id_prefix="L",
        id_to_dsl=id_to_dsl, ref_to_dsl=ref_to_dsl,
    )
    feedback = "\n\n".join(s for s in (structural, logical) if s).strip()

    stc_passed, stc_total = _conformance_ratio(constraints, "STC")
    lsc_passed, lsc_total = _conformance_ratio(constraints, "LSC")

    def ratio(passed: int, total: int) -> float:
        return passed / total if total else 1.0

    metrics = {
        "structural_conformance": {
            "passed": stc_passed, "total": stc_total,
            "ratio": ratio(stc_passed, stc_total),
        },
        "logical_sanity_conformance": {
            "passed": lsc_passed, "total": lsc_total,
            "ratio": ratio(lsc_passed, lsc_total),
        },
    }
    return {
        "grammar": grammar_report,
        "constraints": constraints,
        "feedback": feedback,
        "metrics": metrics,
    }


def variant_just_feedback(feedback_text: str, rootcause_text: str,
                          id_map: dict[str, str]) -> str:
    return feedback_text.strip()


def variant_just_rootcause(feedback_text: str, rootcause_text: str,
                           id_map: dict[str, str]) -> str:
    kept = [
        line for line in rootcause_text.splitlines()
        if not _INCLUDES_RE.match(line)
    ]
    return "\n".join(kept).strip()


def variant_rootcause_feedback_combined(feedback_text: str, rootcause_text: str,
                                        id_map: dict[str, str]) -> str:
    out_lines: list[str] = []
    for line in rootcause_text.splitlines():
        m = _INCLUDES_RE.match(line)
        if not m:
            out_lines.append(line)
            continue
        prefix = m.group(1)
        ids = [tok.strip() for tok in m.group(2).split(",") if tok.strip()]
        out_lines.append(prefix.rstrip())
        for fid in ids:
            desc = id_map.get(fid)
            if desc:
                out_lines.append(f"    - [{fid}] {desc}")
            else:
                out_lines.append(f"    - [{fid}] (referenced error not found in feedback)")
    return "\n".join(out_lines).strip()


VARIANT_BUILDERS = {
    "just_feedback": variant_just_feedback,
    "just_rootcause": variant_just_rootcause,
    "rootcause_feedback_combined": variant_rootcause_feedback_combined,
}


async def refine_sample(dataset: str, llm_name: str, sample_dir: Path, sid: str,
                        pd_dir: Path, llm_caller: LLMCaller,
                        overwrite: bool, max_retries: int = 3) -> dict[str, str]:
    feedback_text = _read(sample_dir / f"{sid}.feedback")
    rootcause_text = _read(sample_dir / f"{sid}.rootcause")
    dsl_model = dsl_from_grammar(sample_dir / f"{sid}.grammar")
    process_description = _read(pd_dir / f"{sid}.txt")
    id_map = build_feedback_id_map(feedback_text)
    source_traces, model_traces = render_traces(sample_dir / f"{sid}.trace")
    input_elements = render_input_elements(sample_dir / f"{sid}.mapping",
                                           process_description)

    async def process_variant(variant: str) -> str:
        out_dir = OUT_ROOT / variant / dataset / llm_name / sid
        refined_path = out_dir / f"{sid}.refined"

        if refined_path.exists() and refined_path.read_text(encoding="utf-8").strip() \
                and not overwrite:
            return "skip"

        if not feedback_text.strip():
            _write_sample_outputs(out_dir, sid, dsl_model, cost_call=None)
            return "empty-feedback"

        feedback = VARIANT_BUILDERS[variant](feedback_text, rootcause_text, id_map)
        prompt = prompts.build_refinement_prompt(process_description, dsl_model, feedback,
                              input_elements=input_elements,
                              source_traces=source_traces, model_traces=model_traces)

        last_problem = "no response"
        for attempt in range(1, max_retries + 1):
            result = await llm_caller.call_async(
                messages=[{"role": "user", "content": prompt}],
                label=f"refine_{variant}",
            )
            if result.error:
                last_problem = result.error
            elif not (result.response or "").strip():
                last_problem = "empty response"
            else:
                _write_sample_outputs(out_dir, sid, result.response,
                                      cost_call=result.to_dict())
                return "ok"
            if attempt < max_retries:
                await asyncio.sleep(2 ** (attempt - 1))

        return f"error: {last_problem}"

    results = await asyncio.gather(*(process_variant(v) for v in VARIANTS))
    return dict(zip(VARIANTS, results))


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def _cost_from_call(call: dict | None) -> dict:
    if call is None:
        return {
            "model_name": None, "reasoning_effort": None, "total_calls": 0,
            "total_errors": 0, "total_prompt_tokens": 0, "total_completion_tokens": 0,
            "total_reasoning_tokens": None, "total_tokens": 0, "total_cost": None,
            "calls": [],
        }
    rt = call.get("reasoning_tokens")
    return {
        "model_name": call.get("model_name"),
        "reasoning_effort": call.get("reasoning_effort"),
        "total_calls": 1,
        "total_errors": 1 if call.get("error") else 0,
        "total_prompt_tokens": call.get("prompt_tokens") or 0,
        "total_completion_tokens": call.get("completion_tokens") or 0,
        "total_reasoning_tokens": rt if isinstance(rt, (int, float)) else None,
        "total_tokens": call.get("total_tokens") or 0,
        "total_cost": call.get("cost"),
        "calls": [call],
    }


def _write_sample_outputs(out_dir: Path, sid: str, refined_dsl: str,
                          cost_call: dict | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_dsl = _strip_code_fence(refined_dsl)
    (out_dir / f"{sid}.refined").write_text(clean_dsl, encoding="utf-8")
    (out_dir / f"{sid}.cost").write_text(
        json.dumps(_cost_from_call(cost_call), indent=2), encoding="utf-8")

    val = validate_refined_dsl(clean_dsl)
    metrics = dict(val["metrics"])
    (out_dir / f"{sid}.grammar").write_text(
        json.dumps(val["grammar"], indent=2), encoding="utf-8")
    (out_dir / f"{sid}.constraints").write_text(
        json.dumps(val["constraints"], indent=2), encoding="utf-8")
    (out_dir / f"{sid}.feedback").write_text(val["feedback"], encoding="utf-8")
    (out_dir / f"{sid}.metrics").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")


def family_from_llm_name(llm_name: str) -> str:
    base = llm_name.split("-", 1)[0].lower()
    if base in ("gemini","qwen"):
        return base
    raise ValueError(f"Cannot infer model family from '{llm_name}'; pass --model.")


def list_run_folders(dataset: str) -> list[str]:
    ddir = RESULTS / dataset
    if not ddir.is_dir():
        return []
    return sorted(p.name for p in ddir.iterdir() if p.is_dir())


def resolve_llm_arg(dataset: str, llm_arg: str | None) -> list[str]:
    all_folders = list_run_folders(dataset)
    if not llm_arg:
        return all_folders
    if llm_arg in all_folders:
        return [llm_arg]
    return [f for f in all_folders if f.startswith(f"{llm_arg}-")]


async def run_family(dataset: str, llm_name: str, model_key: str | None,
                     reasoning_effort, diag_filter: str | None, limit: int | None,
                     overwrite: bool) -> dict[str, int]:
    llm_dir = RESULTS / dataset / llm_name
    if not llm_dir.is_dir():
        print(f"[skip] {llm_dir} not found")
        return {"ok": 0, "empty": 0, "skip": 0, "failed": 0}

    family = model_key or family_from_llm_name(llm_name)
    llm_caller = LLMCaller(model=family, reasoning_effort=reasoning_effort)
    pd_dir = HERE / dataset / "GT" / "PD"

    sample_dirs = sorted(
        (p for p in llm_dir.iterdir()
         if p.is_dir() and (p / f"{p.name}.feedback").exists()),
        key=lambda p: (len(p.name), p.name),
    )
    if diag_filter:
        sample_dirs = [p for p in sample_dirs if p.name == diag_filter]
    if limit is not None:
        sample_dirs = sample_dirs[:limit]

    total = len(sample_dirs)
    print(f"=== {dataset} / {llm_name}  (model='{llm_caller.model_name}')  {total} samples ===",
          flush=True)

    counts = {"ok": 0, "empty": 0, "skip": 0, "failed": 0}
    start = time.perf_counter()
    for index, sample_dir in enumerate(sample_dirs, start=1):
        sid = sample_dir.name
        statuses = await refine_sample(
            dataset, llm_name, sample_dir, sid, pd_dir, llm_caller,
            overwrite,
        )
        for st in statuses.values():
            if st == "ok":
                counts["ok"] += 1
            elif st == "empty-feedback":
                counts["empty"] += 1
            elif st == "skip":
                counts["skip"] += 1
            else:
                counts["failed"] += 1
        summary = ", ".join(f"{v}:{s}" for v, s in statuses.items())
        print(f"[{index}/{total}] {dataset}/{llm_name} {sid}: {summary}", flush=True)

    elapsed = time.perf_counter() - start
    print(f"--- {dataset}/{llm_name}: ok={counts['ok']} empty={counts['empty']} "
          f"skip={counts['skip']} failed={counts['failed']} in {elapsed:.1f}s ---\n",
          flush=True)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description="Refine process models from feedback.")
    ap.add_argument("--dataset", choices=DATASETS,
                    help="Dataset to process. Omit to process all datasets.")
    ap.add_argument("--llm", help="run folder ('gemini-3'), family ('gemini' -> "
                                   "gemini-1..5), or omit for every run folder.")
    ap.add_argument("--model", choices=["qwen", "gemini"],
                    help="override refiner LLM (default: inferred from --llm name).")
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--diag", help="only process this diagram id (e.g. 135).")
    ap.add_argument("--limit", type=int, help="cap number of samples per run folder.")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo samples that already have a non-empty .refined.")
    args = ap.parse_args()

    datasets = [args.dataset] if args.dataset else DATASETS

    grand = {"ok": 0, "empty": 0, "skip": 0, "failed": 0}
    overall_start = time.perf_counter()
    for dataset in datasets:
        families = resolve_llm_arg(dataset, args.llm)
        if not families:
            print(f"No run folders under {RESULTS / dataset}"
                  + (f" matching --llm '{args.llm}'" if args.llm else ""))
            continue
        print(f"[{dataset}] run folders: {', '.join(families)}")
        for llm_name in families:
            counts = asyncio.run(run_family(
                dataset, llm_name, args.model, args.reasoning_effort,
                args.diag, args.limit, args.overwrite,
            ))
            for k in grand:
                grand[k] += counts.get(k, 0)

    elapsed = time.perf_counter() - overall_start
    print(f"=== TOTAL: ok={grand['ok']} empty={grand['empty']} "
          f"skip={grand['skip']} failed={grand['failed']} in {elapsed:.1f}s ===")
    return 1 if grand["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
