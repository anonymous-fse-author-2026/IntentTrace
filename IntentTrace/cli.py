import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .grammar import format_rejected, parse_process
from .llm import DEFAULT_MODEL, LLMCaller, MODELS, resolve_model
from .constraint_checker import ConstraintChecking
from .evaluation import (
    compute_metrics,
    evaluate_traces_with_mapping,
    format_consolidated_feedback,
    format_counter_examples,
    format_metrics,
)
from .parsed_model import build_parsed_model
from .element_matching import format_mapping_details, match_elements
from .trace_extraction import build_traces_holistic, get_all_model_traces
from .actor_var_verification import (
    format_var_actor_validation,
    validate_variables_and_actors,
)
from Render.csv_to_plantuml import (
    FLOW_LAYOUT_HORIZONTAL,
    csv_to_plantuml_final,
    json_to_csv,
)

load_dotenv(Path(__file__).resolve().parent / ".env")

SUFFIXES = (
    "json", "grammar", "constraints", "actorvars", "mapping", "trace",
    "evaluation", "feedback", "metrics", "puml", "cost",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="IntentTrace",
        description="Verify a process model against the formal grammar and the source process text.",
    )
    p.add_argument("model_file", type=Path, metavar="MODEL_FILE",
                   help="Process-model statements to verify.")
    p.add_argument("input", type=Path, metavar="TEXT_FILE",
                   help="Source process-text file.")
    p.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL,
                   help=f"Model name or id. Default: {DEFAULT_MODEL}.")
    p.add_argument("-o", "--name", type=Path,
                   help="Base output name/path without extension. Default: MODEL_FILE stem.")
    p.add_argument("--no-var-actor", action="store_true",
                   help="Disable variable/actor (SMC6/SMC7) validation.")
    p.add_argument("--spec", type=Path, metavar="JSON_FILE",
                   help="Existing spec to reuse instead of segmenting the source text again. "
                        "Accepts a prior .json (reads 'input_elements'), a prior .mapping "
                        "(reads 'spec_elements'), or {'spec_elements': [...], "
                        "'spec_traces': [...]}. Spec traces are taken from --spec-traces, or "
                        "from this file's 'spec_traces'/'source_text_traces' if present.")
    p.add_argument("--spec-traces", type=Path, metavar="JSON_FILE",
                   help="Spec traces to reuse, e.g. a prior .trace file "
                        "(reads 'source_text_traces'). Only used with --spec.")
    p.add_argument("-v", "--verbose", action="store_true", help="Print step-by-step progress.")
    return p.parse_args()


def output_paths(args: argparse.Namespace) -> Dict[str, Path]:
    base = Path(args.name or args.model_file.stem)
    base = base.with_suffix("") if base.suffix else base
    return {s: base.with_suffix(f".{s}") for s in SUFFIXES}


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, file=sys.stderr)


def _model_for_output(model: Dict[str, Any]) -> Dict[str, Any]:
    clean = dict(model)
    clean["relationships"] = [
        {k: v for k, v in rel.items() if k != "condition_ast"}
        for rel in model.get("relationships", [])
    ]
    return clean


def _input_trace_elements(mapping: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
    spans: List[Tuple[int, int, str]] = []
    seen: set[str] = set()
    for category in ("matches", "mutations", "omissions"):
        for entry in mapping.get(category, []):
            for raw in entry.get("verbatim_texts", []):
                span = str(raw).strip()
                if not span or span in seen:
                    continue
                seen.add(span)
                pos = text.find(span)
                if pos < 0:
                    pos = text.lower().find(span.lower())
                if pos < 0:
                    pos = sys.maxsize
                spans.append((pos, len(spans), span))
    spans.sort(key=lambda item: (item[0], item[1]))
    return [{"id": f"i{n + 1}", "content": s} for n, (_, _, s) in enumerate(spans)]


def _load_spec(path: Optional[Path], traces_path: Optional[Path]):
    if path is None:
        return None, None

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"{path}: expected a JSON object.")

    elements = None
    for key in ("spec_elements", "input_elements"):
        if isinstance(data.get(key), list):
            elements = data[key]
            break
    if elements is None:
        raise RuntimeError(
            f"{path}: found no spec elements (looked for 'spec_elements' and 'input_elements')."
        )

    traces = None
    trace_source = data
    if traces_path is not None:
        loaded = json.loads(traces_path.read_text(encoding="utf-8"))
        trace_source = loaded if isinstance(loaded, dict) else {}
        if isinstance(loaded, list):
            traces = loaded
    if traces is None:
        for key in ("spec_traces", "source_text_traces"):
            if isinstance(trace_source.get(key), list):
                traces = trace_source[key]
                break

    return elements, traces


def _print_spec_revisions(result: Any) -> None:
    if not getattr(result, "revised_spec", False):
        return
    print("\n=== SPEC REVISIONS ===")
    revisions = result.spec_revisions or []
    if not revisions:
        print("Spec reused unchanged.")
        return
    for entry in revisions:
        ids = ", ".join(entry.get("ids", []))
        reason = str(entry.get("reason", "")).strip()
        line = f"{entry.get('action', '?')}: {ids}"
        print(f"{line} — {reason}" if reason else line)


def _print_checks(validation: Dict[str, Any], prefix: str) -> None:
    keys = sorted(
        (k for k in validation if k.startswith(prefix) and "_" not in k),
        key=lambda k: int(k[3:]),
    )
    for k in keys:
        print(f"{k}: {validation[k]}")
    print(f"all_{prefix.lower()}_passed: {validation.get(f'all_{prefix.lower()}_passed', False)}")


def _parse_gate(args: argparse.Namespace, out: Dict[str, Path], verbose: bool):
    report = parse_process(args.model_file.read_text(encoding="utf-8"))
    print("=== GRAMMAR VERIFICATION ===")
    print(f"Accepted statements: {len(report.accepted)}")
    print(format_rejected(report))
    if report.rejected:
        print(
            "WARNING: some statements did not conform to the grammar and were not parsed.",
            file=sys.stderr,
        )
    out["grammar"].parent.mkdir(parents=True, exist_ok=True)
    out["grammar"].write_text(
        json.dumps(
            {
                "ok": report.ok,
                "accepted_count": len(report.accepted),
                "rejected_count": len(report.rejected),
                "accepted": report.accepted,
                "rejected": report.rejected,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _log(verbose, f"Grammar report written to: {out['grammar']}")
    if not report.model.get("elements"):
        raise RuntimeError("No conforming node statements were found to verify.")
    return report.model


def _render_puml(model: Dict[str, Any], path: Path, verbose: bool) -> None:
    path.write_text(
        csv_to_plantuml_final(json_to_csv(model, flow_layout=FLOW_LAYOUT_HORIZONTAL)),
        encoding="utf-8",
    )
    _log(verbose, f"PlantUML written to: {path}")


def _actorvars_report(result: Any) -> Dict[str, Any]:
    d = result.to_dict()
    error = d.get("error")
    smc6 = result.smc6_passed if not error else None
    smc7 = result.smc7_passed if not error else None
    all_passed = None if error else (smc6 and smc7)
    return {
        "evaluation_passed": all_passed,
        "engine": "llm",
        "llm_smc_engine": {
            "model_id": d.get("model_id"),
            "reasoning_effort": d.get("reasoning_effort"),
            "error": error,
        },
        "SMC6": {"name": "variable definitions", "passed": smc6, "details": d["SMC6"]},
        "SMC7": {"name": "actor groupings", "passed": smc7, "details": d["SMC7"]},
        "smc_count": 2,
        "all_smc_passed": all_passed,
    }


async def main() -> int:
    args = parse_args()

    try:
        process_text = args.input.read_text(encoding="utf-8")
        out = output_paths(args)
        model_id = MODELS[resolve_model(args.model)]["model"]
        effort = str(MODELS[resolve_model(model_id)].get("reasoning_effort") or "none")
        _log(args.verbose, f"[config] model: {model_id}, effort: {effort}")

        llm = LLMCaller(model=model_id)
        model = _parse_gate(args, out, args.verbose)
        parsed = build_parsed_model(model)

        given_spec, given_spec_traces = _load_spec(args.spec, args.spec_traces)
        if given_spec is not None:
            _log(args.verbose,
                 f"[spec] reusing {len(given_spec)} element(s) and "
                 f"{len(given_spec_traces or [])} trace(s) from {args.spec}")

        validation = ConstraintChecking(diagram_json=model).validate()
        print("\n=== STRUCTURAL VALIDATION ===")
        _print_checks(validation, "STC")
        print("\n=== LOGICAL SANITY VALIDATION ===")
        print(f"engine: {validation.get('logical_sanity_validation_engine', 'sampler')}")
        _print_checks(validation, "LSC")

        async def matching_then_traces():
            mapping = await match_elements(
                input_text=process_text, model=parsed, llm_caller=llm,
                model_id=model_id, reasoning_effort=effort,
                spec_elements=given_spec, spec_traces=given_spec_traces,
            )
            if mapping.revised_spec:
                return mapping, mapping.spec_elements, mapping.spec_traces or []

            elements = _input_trace_elements(mapping.to_dict(), process_text)
            traces = []
            if elements:
                result = await build_traces_holistic(
                    llm_caller=llm, elements=elements, original_text=process_text
                )
                traces = result.get("traces", [])
            return mapping, elements, traces

        async def var_actor():
            if args.no_var_actor:
                return None
            return await validate_variables_and_actors(
                process_text=process_text, model=parsed, llm_caller=llm,
                model_id=model_id, reasoning_effort=effort,
            )

        started = time.perf_counter()
        (mapping, elements, input_traces), var_actor_result = await asyncio.gather(
            matching_then_traces(), var_actor()
        )
        wall_clock = time.perf_counter() - started

        print("\n=== ELEMENT MAPPING ===")
        if args.verbose:
            print(format_mapping_details(mapping))

        _print_spec_revisions(mapping)

        if not args.no_var_actor:
            print("\n=== VARIABLE & ACTOR VALIDATION ===")
            print(format_var_actor_validation(var_actor_result))

        print("\n=== MODEL EXECUTION TRACES (INIT → END) ===")
        model_traces = get_all_model_traces(parsed).get("traces", [])
        for n, trace in enumerate(model_traces, 1):
            print(f"Trace {n}: {' -> '.join(trace)}")
        if not model_traces:
            print("No init-to-end traces found in the model.")

        print("\n=== SOURCE TEXT ELEMENTS ===")
        model["input_elements"] = elements
        for element in elements:
            print(f"{element['id']}: {str(element['content']).strip()}")
        if not elements:
            print("No verbatim text elements were mapped.")

        print("\n=== SOURCE TEXT EXECUTION TRACES ===")
        for n, trace in enumerate(input_traces, 1):
            print(f"Trace {n}: {' -> '.join(trace)}")
        if not input_traces:
            print("No traces were produced from mapped source-text elements.")

        print("\n=== TRACE EVALUATION ===")
        evaluation = evaluate_traces_with_mapping(
            input_traces=input_traces,
            gen_traces=model_traces,
            mapping_result=mapping,
            input_trace_elements=elements,
            gen_elements=model.get("elements", []),
        )
        print(format_counter_examples(evaluation))

        feedback = format_consolidated_feedback(
            mapping_result=mapping,
            evaluation_result=evaluation,
            var_actor_result=var_actor_result,
            constraints=validation,
            model=model,
            input_trace_elements=elements,
        )
        out["feedback"].write_text(feedback, encoding="utf-8")
        _log(args.verbose, f"Feedback written to: {out['feedback']}")

        clean = _model_for_output(model)

        metrics = compute_metrics(
            evaluation_result=evaluation,
            mapping_result=mapping,
            constraints=validation,
            var_actor_result=var_actor_result,
        )
        print("\n" + format_metrics(metrics))

        written = {
            "metrics": metrics,
            "evaluation": evaluation.to_dict(),
            "json": clean,
            "trace": {"model_traces": model_traces, "source_text_traces": input_traces},
            "mapping": mapping.to_dict(),
            "constraints": validation,
        }
        if not args.no_var_actor:
            written["actorvars"] = _actorvars_report(var_actor_result)

        for key, payload in written.items():
            out[key].write_text(json.dumps(payload, indent=2), encoding="utf-8")
            _log(args.verbose, f"{key} written to: {out[key]}")

        _render_puml(clean, out["puml"], args.verbose)

        cost = llm.get_cost_summary()
        out["cost"].write_text(json.dumps(cost, indent=2), encoding="utf-8")
        summed = sum(c.get("duration_seconds") or 0.0 for c in cost["calls"])
        print(
            f"\n=== LLM COST ===\n"
            f"calls: {cost['total_calls']}  tokens: {cost['total_tokens']}  "
            f"cost: {cost['total_cost']}  wall-clock: {wall_clock:.2f}s  "
            f"(summed call time: {summed:.2f}s)"
        )

        print("\n=== PARSED MODEL JSON ===")
        print(json.dumps(clean, indent=2))
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
