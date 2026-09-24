CREDIT_CLEAN = 1.0
CREDIT_MUTATED = 0.5
CREDIT_REORDERED = 0.5


_COST_CLEAN = 0.0
_COST_MUTATED = 0.5
_COST_BREAK = 1.0

import json
import re
from typing import List, Dict, Any, Tuple, Optional, Set

import numpy as np


def _to_json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_json_safe(v) for v in obj)
    return obj


def _extract_e_ids(ref: str) -> List[str]:
    ref = str(ref)
    return re.findall(r"\b[inbce]\d+\b", ref)


def _content_of(elem_id: str, elements: Dict[str, Dict[str, Any]]) -> str:
    elem = elements.get(elem_id)
    if not elem:
        return str(elem_id)
    text = elem.get("main_content") or elem.get("content")
    return str(text) if text else str(elem_id)


def _trace_content(trace_ids: List[str], elements: Dict[str, Dict[str, Any]]) -> List[str]:
    return [_content_of(elem_id, elements) for elem_id in trace_ids]


def _build_mapping_pairs_from_result(
    mapping_dict: Any,
    input_trace_elements: List[Dict[str, Any]],
) -> List[Tuple[str, str]]:
    if hasattr(mapping_dict, "to_dict"):
        mapping_dict = mapping_dict.to_dict()
    if not isinstance(mapping_dict, dict):
        return []

    text_to_input_id = {
        str(elem.get("content", "")).strip(): str(elem.get("id", "")).strip()
        for elem in input_trace_elements
        if str(elem.get("content", "")).strip() and str(elem.get("id", "")).strip()
    }

    pairs: Set[Tuple[str, str]] = set()
    for category in ("matches", "mutations"):
        for entry in mapping_dict.get(category, []):
            gen_ids: Set[str] = set()
            for ref in entry.get("refs", []):
                gen_ids.update(_extract_e_ids(ref))
            for raw_text in entry.get("verbatim_texts", []):
                input_id = text_to_input_id.get(str(raw_text).strip())
                if not input_id:
                    continue
                for gen_id in gen_ids:
                    pairs.add((input_id, gen_id))
    return list(pairs)


def _mutated_gen_ids_from_result(mapping_dict: Any) -> Set[str]:
    if hasattr(mapping_dict, "to_dict"):
        mapping_dict = mapping_dict.to_dict()
    if not isinstance(mapping_dict, dict):
        return set()
    mutated: Set[str] = set()
    for entry in mapping_dict.get("mutations", []) or []:
        if isinstance(entry, dict):
            for ref in entry.get("refs", []):
                mutated.update(_extract_e_ids(ref))
    return mutated


def align_traces(
    input_trace: List[str],
    gen_trace: List[str],
    input_to_gen: Dict[str, Set[str]],
    input_elements: Dict[str, Dict[str, Any]],
    gen_elements: Dict[str, Dict[str, Any]],
    mutated_gen_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    mutated_gen_ids = mutated_gen_ids or set()
    n, m = len(input_trace), len(gen_trace)

    def maps(i: int, j: int) -> bool:
        return gen_trace[j] in input_to_gen.get(input_trace[i], set())

    def is_mutated(j: int) -> bool:
        return gen_trace[j] in mutated_gen_ids

    def match_cost(i: int, j: int) -> Optional[float]:
        if not maps(i, j):
            return None
        return _COST_MUTATED if is_mutated(j) else _COST_CLEAN


    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i * _COST_BREAK
    for j in range(1, m + 1):
        cost[0][j] = j * _COST_BREAK
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            mc = match_cost(i - 1, j - 1)
            best = min(cost[i - 1][j] + _COST_BREAK, cost[i][j - 1] + _COST_BREAK)
            if mc is not None:
                best = min(best, cost[i - 1][j - 1] + mc)
            cost[i][j] = best


    matched_input: Set[int] = set()
    matched_gen: Set[int] = set()
    match_input_to_gen: Dict[int, int] = {}
    mutated_matched_gen: Set[int] = set()
    omissions: List[int] = []
    additions: List[int] = []

    i, j = n, m
    while i > 0 or j > 0:
        mc = match_cost(i - 1, j - 1) if (i > 0 and j > 0) else None
        if mc is not None and cost[i][j] == cost[i - 1][j - 1] + mc:
            matched_input.add(i - 1)
            matched_gen.add(j - 1)
            match_input_to_gen[i - 1] = j - 1
            if is_mutated(j - 1):
                mutated_matched_gen.add(j - 1)
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + _COST_BREAK:
            omissions.append(i - 1)
            i -= 1
        else:
            additions.append(j - 1)
            j -= 1
    omissions.reverse()
    additions.reverse()

    divergences, compacted, claimed = _classify_divergences(
        omissions,
        additions,
        matched_gen,
        match_input_to_gen,
        input_trace,
        gen_trace,
        input_to_gen,
        input_elements,
        gen_elements,
    )


    matched_input |= compacted
    matched_gen |= claimed


    reordered_gen = {
        gid for div in divergences if div["type"] == "reordering"
        for gid in div.get("actual", {}).get("ids", [])
    }
    reordered_input = {
        iid for div in divergences if div["type"] == "reordering"
        for iid in div.get("expected", {}).get("ids", [])
    }
    source_credit = 0.0
    for idx in matched_input:
        if input_trace[idx] in reordered_input:
            source_credit += CREDIT_REORDERED
        else:
            source_credit += CREDIT_CLEAN
    model_credit = 0.0
    n_clean = n_mutated = 0
    for idx in matched_gen:
        if gen_trace[idx] in reordered_gen:
            model_credit += CREDIT_REORDERED
        elif idx in mutated_matched_gen:
            model_credit += CREDIT_MUTATED
            n_mutated += 1
        else:
            model_credit += CREDIT_CLEAN
            n_clean += 1


    source_credit -= CREDIT_MUTATED * min(
        n_mutated, sum(1 for idx in matched_input if input_trace[idx] not in reordered_input)
    )
    n_reordered = sum(1 for div in divergences if div["type"] == "reordering")


    if n == 0 and m == 0:
        similarity = 1.0
    elif n == 0 or m == 0:
        similarity = 0.0
    else:
        recall = source_credit / n
        precision = model_credit / m
        similarity = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "matched_input_steps": len(matched_input),
        "matched_gen_steps": len(matched_gen),
        "matched_input_ids": [input_trace[idx] for idx in sorted(matched_input)],
        "matched_gen_ids": [gen_trace[idx] for idx in sorted(matched_gen)],
        "input_len": n,
        "gen_len": m,
        "divergences": divergences,
        "similarity": similarity,

        "source_credit": source_credit,
        "model_credit": model_credit,
        "clean_steps": n_clean,
        "mutated_steps": n_mutated,
        "reordered_steps": n_reordered,


        "forgiven_scaffolding_steps": _forgiven_scaffolding_credit(
            divergences, gen_elements
        ),
    }


_STRUCTURAL_TYPES = {"init", "end", "branch", "converge"}


_SCORE_FORGIVEN_TYPES = {"init", "end", "converge"}


def _is_structural_addition(j: int, gen_trace: List[str], covered: Set[int],
                            gen_elements: Dict[str, Dict[str, Any]]) -> bool:
    gen_id = gen_trace[j]
    node_type = str(gen_elements.get(gen_id, {}).get("type", "")).strip().lower()
    if node_type not in _STRUCTURAL_TYPES:
        return False

    prev_covered = j == 0 or (j - 1) in covered
    next_covered = j == len(gen_trace) - 1 or (j + 1) in covered
    return prev_covered and next_covered


def _forgiven_scaffolding_credit(
    divergences: List[Dict[str, Any]],
    gen_elements: Dict[str, Dict[str, Any]],
) -> int:
    n = 0
    for div in divergences:
        if str(div.get("type", "")).strip().lower() != "addition":
            continue
        actual = div.get("actual", {}) or {}

        types = [str(t).strip().lower() for t in (actual.get("types") or [])]
        if not types:
            types = [
                str(gen_elements.get(gid, {}).get("type", "")).strip().lower()
                for gid in (actual.get("ids") or [])
            ]
        n += sum(1 for t in types if t in _SCORE_FORGIVEN_TYPES)
    return n


def _classify_divergences(
    omissions: List[int],
    additions: List[int],
    matched_gen: Set[int],
    match_input_to_gen: Dict[int, int],
    input_trace: List[str],
    gen_trace: List[str],
    input_to_gen: Dict[str, Set[str]],
    input_elements: Dict[str, Dict[str, Any]],
    gen_elements: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Set[int]]:
    addition_by_gen_id = {gen_trace[j]: j for j in additions}
    matched_gen_ids = {gen_trace[j] for j in matched_gen}


    claimed_gen_ids: Set[str] = set()
    for i_idx in match_input_to_gen:
        claimed_gen_ids |= input_to_gen.get(input_trace[i_idx], set())
    used_additions: Set[int] = set()
    compacted: Set[int] = set()
    claimed: Set[int] = set()
    divergences: List[Dict[str, Any]] = []

    def step_crossed_over(input_idx: int, moved_gen_ids: List[str]) -> Optional[str]:
        moved_positions = [gen_trace.index(g) for g in moved_gen_ids if g in gen_trace]
        if not moved_positions:
            return None
        moved_pos = moved_positions[0]


        prev_gen = next(
            (match_input_to_gen[k] for k in range(input_idx - 1, -1, -1) if k in match_input_to_gen),
            None,
        )
        next_gen = next(
            (match_input_to_gen[k] for k in range(input_idx + 1, len(input_trace)) if k in match_input_to_gen),
            None,
        )


        if prev_gen is not None and moved_pos < prev_gen:
            return gen_trace[prev_gen]
        if next_gen is not None and moved_pos > next_gen:
            return gen_trace[next_gen]
        return None

    for i in omissions:
        in_id = input_trace[i]
        allowed_gen_ids = input_to_gen.get(in_id, set())
        moved = [
            (gen_id, addition_by_gen_id[gen_id])
            for gen_id in allowed_gen_ids
            if gen_id in addition_by_gen_id and addition_by_gen_id[gen_id] not in used_additions
        ]
        if moved:
            for _, j in moved:
                used_additions.add(j)
            moved_gen_ids = [gen_id for gen_id, _ in moved]
            reordering: Dict[str, Any] = {
                "type": "reordering",
                "description": "Step appears in a different position",
                "expected": {
                    "ids": [in_id],
                    "content": [_content_of(in_id, input_elements)],
                },
                "actual": {
                    "ids": moved_gen_ids,
                    "content": [_content_of(gen_id, gen_elements) for gen_id in moved_gen_ids],
                },
            }
            swapped_gen_id = step_crossed_over(i, moved_gen_ids)
            if swapped_gen_id and swapped_gen_id not in moved_gen_ids:
                reordering["swapped_with"] = {
                    "ids": [swapped_gen_id],
                    "content": [_content_of(swapped_gen_id, gen_elements)],
                }
            divergences.append(reordering)
        elif allowed_gen_ids & matched_gen_ids:


            compacted.add(i)
        else:
            divergences.append({
                "type": "omission",
                "description": "Step missing from the model trace",
                "expected": {
                    "ids": [in_id],
                    "content": [_content_of(in_id, input_elements)],
                },
            })

    covered = matched_gen | used_additions
    for j in additions:
        if j in used_additions:
            continue
        gen_id = gen_trace[j]


        if gen_id in claimed_gen_ids:
            claimed.add(j)
            continue
        if _is_structural_addition(j, gen_trace, covered | claimed, gen_elements):


            claimed.add(j)
            continue
        divergences.append({
            "type": "addition",
            "description": "Extra step in the model trace",
            "actual": {
                "ids": [gen_id],
                "content": [_content_of(gen_id, gen_elements)],
                "types": [str(gen_elements.get(gen_id, {}).get("type", "")).strip()],
            },
        })

    return divergences, compacted, claimed


class TraceEvaluationResult:
    def __init__(self, completeness, correctness, trace_matches, counter_examples, metrics):
        self.compound_completeness = completeness
        self.compound_correctness = correctness
        self.trace_matches = trace_matches
        self.counter_examples = counter_examples
        self.detailed_metrics = metrics

    def to_dict(self):


        return {
            "trace_matches": _to_json_safe(self.trace_matches),
            "counter_examples": _to_json_safe(self.counter_examples),
        }


def evaluate_traces(
    input_traces: List[List[str]],
    gen_traces: List[List[str]],
    mapping_pairs: List[Tuple[str, str]],
    input_elements: List[Dict[str, Any]],
    gen_elements: List[Dict[str, Any]],
    similarity_threshold: float = 0.5,
    mutated_gen_ids: Optional[Set[str]] = None,
) -> TraceEvaluationResult:
    input_elem_dict = {e.get("id"): e for e in input_elements}
    gen_elem_dict = {e.get("id"): e for e in gen_elements}
    mutated_gen_ids = mutated_gen_ids or set()

    input_to_gen: Dict[str, Set[str]] = {}
    for in_id, gen_id in mapping_pairs:
        input_to_gen.setdefault(in_id, set()).add(gen_id)

    n_in, n_gen = len(input_traces), len(gen_traces)

    if n_in == 0 and n_gen == 0:
        return TraceEvaluationResult(1.0, 1.0, [], [], {})
    if n_in == 0:
        return TraceEvaluationResult(0.0, 0.0, [], [{"type": "hallucinated", "count": n_gen}], {"total_gen": n_gen})
    if n_gen == 0:
        return TraceEvaluationResult(0.0, 1.0, [], [{"type": "missing", "count": n_in}], {"total_in": n_in})

    similarity = np.zeros((n_in, n_gen))
    alignments: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for i in range(n_in):
        for j in range(n_gen):
            res = align_traces(
                input_traces[i], gen_traces[j], input_to_gen, input_elem_dict, gen_elem_dict,
                mutated_gen_ids,
            )
            alignments[(i, j)] = res
            similarity[i, j] = res["similarity"]


    def best_against(fixed_is_input: bool, idx: int) -> Tuple[int, int]:
        candidates = range(n_gen) if fixed_is_input else range(n_in)
        if fixed_is_input:
            key = lambda k: -similarity[idx, k]
        else:
            key = lambda k: -similarity[k, idx]
        best = min(candidates, key=key)
        return (idx, best) if fixed_is_input else (best, idx)

    pairs: List[Tuple[int, int]] = []
    if n_in <= n_gen:
        pairs = [best_against(True, i) for i in range(n_in)]
    else:
        pairs = [best_against(False, j) for j in range(n_gen)]

    trace_matches = []
    matched_inputs: Set[int] = set()
    matched_gens: Set[int] = set()
    covered_input_steps = 0
    covered_gen_steps = 0

    source_credit = 0.0
    model_credit = 0.0
    clean_steps = mutated_steps = reordered_steps = 0

    forgiven_scaffolding_steps = 0

    for r, c in pairs:
        score = similarity[r, c]
        if score < similarity_threshold:
            continue
        res = alignments[(r, c)]
        matched_inputs.add(r)
        matched_gens.add(c)
        covered_input_steps += res["matched_input_steps"]
        covered_gen_steps += res["matched_gen_steps"]
        source_credit += res["source_credit"]
        model_credit += res["model_credit"]
        clean_steps += res["clean_steps"]
        mutated_steps += res["mutated_steps"]
        reordered_steps += res["reordered_steps"]
        forgiven_scaffolding_steps += res["forgiven_scaffolding_steps"]
        trace_matches.append({
            "input_trace_idx": r,
            "gen_trace_idx": c,
            "similarity": float(score),
            "matched_input_ids": res["matched_input_ids"],
            "matched_gen_ids": res["matched_gen_ids"],
            "input_len": res["input_len"],
            "gen_len": res["gen_len"],
            "input_trace": {
                "ids": input_traces[r],
                "content": _trace_content(input_traces[r], input_elem_dict),
            },
            "gen_trace": {
                "ids": gen_traces[c],
                "content": _trace_content(gen_traces[c], gen_elem_dict),
            },
            "divergences": res["divergences"],
        })

    counter_examples: List[Dict[str, Any]] = []
    for match in trace_matches:
        if match["divergences"]:
            counter_examples.append({
                "type": "partial_mismatch",
                "input_trace_idx": match["input_trace_idx"],
                "gen_trace_idx": match["gen_trace_idx"],
                "similarity": match["similarity"],
                "divergences": match["divergences"],
            })
    for i in range(n_in):
        if i not in matched_inputs:
            counter_examples.append({
                "type": "missing_trace",
                "input_trace_idx": i,
                "expected_trace": {
                    "ids": input_traces[i],
                    "content": _trace_content(input_traces[i], input_elem_dict),
                },
            })
    for j in range(n_gen):
        if j not in matched_gens:
            counter_examples.append({
                "type": "hallucinated_trace",
                "gen_trace_idx": j,
                "actual_trace": {
                    "ids": gen_traces[j],
                    "content": _trace_content(gen_traces[j], gen_elem_dict),
                },
            })

    total_input_steps = sum(len(t) for t in input_traces)
    total_gen_steps = sum(len(t) for t in gen_traces)
    completeness = covered_input_steps / total_input_steps if total_input_steps else 0.0
    correctness = covered_gen_steps / total_gen_steps if total_gen_steps else 0.0

    metrics = {
        "total_input_traces": n_in,
        "total_gen_traces": n_gen,
        "matched_pairs": len(trace_matches),
        "total_input_steps": total_input_steps,
        "total_gen_steps": total_gen_steps,


        "trace_source_credit": source_credit,
        "trace_model_credit": model_credit,
        "clean_steps": clean_steps,
        "mutated_steps": mutated_steps,
        "reordered_steps": reordered_steps,


        "forgiven_scaffolding_steps": forgiven_scaffolding_steps,
    }
    return TraceEvaluationResult(completeness, correctness, trace_matches, counter_examples, metrics)


def evaluate_traces_with_mapping(
    input_traces: List[List[str]],
    gen_traces: List[List[str]],
    mapping_result: Any,
    input_trace_elements: List[Dict[str, Any]],
    gen_elements: List[Dict[str, Any]],
    similarity_threshold: float = 0.5,
) -> TraceEvaluationResult:
    mapping_pairs = _build_mapping_pairs_from_result(mapping_result, input_trace_elements)
    mutated_gen_ids = _mutated_gen_ids_from_result(mapping_result)
    return evaluate_traces(
        input_traces=input_traces,
        gen_traces=gen_traces,
        mapping_pairs=mapping_pairs,
        input_elements=input_trace_elements,
        gen_elements=gen_elements,
        similarity_threshold=similarity_threshold,
        mutated_gen_ids=mutated_gen_ids,
    )


def format_counter_examples(result: TraceEvaluationResult, max_examples: int = 10) -> str:
    output = "\n### Trace Analysis\n\n"

    if result.trace_matches:
        output += f"## Matched Trace Pairs ({len(result.trace_matches)})\n\n"
        for idx, match in enumerate(result.trace_matches[:max_examples], 1):
            output += f"### Match {idx}: Input {match['input_trace_idx']} <-> Gen {match['gen_trace_idx']}\n"
            output += f"**Similarity:** {match['similarity']:.1%}\n\n"

            input_content = match["input_trace"]["content"]
            gen_content = match["gen_trace"]["content"]
            output += "**Input Trace:** " + (" -> ".join(f"`{c}`" for c in input_content) or "_(empty)_") + "\n\n"
            output += "**Model Trace:** " + (" -> ".join(f"`{c}`" for c in gen_content) or "_(empty)_") + "\n\n"

            divs = match["divergences"]
            if not divs:
                output += "_Perfect alignment._\n\n---\n"
                continue

            output += "**Differences:**\n"
            for n, d in enumerate(divs, 1):
                if d["type"] == "omission":
                    text = " -> ".join(d["expected"]["content"])
                    output += f"{n}. Omission: `{text}` missing from the model trace\n"
                elif d["type"] == "addition":
                    text = " -> ".join(d["actual"]["content"])
                    output += f"{n}. Addition: model added `{text}`\n"
                elif d["type"] == "reordering":
                    exp = " -> ".join(d["expected"]["content"])
                    act = " -> ".join(d["actual"]["content"])
                    line = f"{n}. Reordering: `{exp}` expected here, found `{act}` elsewhere"
                    if d.get("swapped_with"):
                        line += f" (now in its place: `{' -> '.join(d['swapped_with']['content'])}`)"
                    output += line + "\n"
            output += "\n---\n"

    missing = [c for c in result.counter_examples if c.get("type") == "missing_trace"]
    if missing:
        output += f"\n## Missing Input Traces ({len(missing)})\n"
        for ce in missing[:5]:
            text = " -> ".join(ce["expected_trace"]["content"])
            output += f"- Input {ce['input_trace_idx']}: `{text}`\n"

    hallucinated = [c for c in result.counter_examples if c.get("type") == "hallucinated_trace"]
    if hallucinated:
        output += f"\n## Hallucinated Model Traces ({len(hallucinated)})\n"
        for ce in hallucinated[:5]:
            text = " -> ".join(ce["actual_trace"]["content"])
            output += f"- Gen {ce['gen_trace_idx']}: `{text}`\n"

    return output


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _join_path_text(values: Any) -> str:
    return " -> ".join(str(v).strip() for v in _as_list(values) if str(v).strip())


def _dedup_key(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


_STC_DESCRIPTIONS: Dict[str, str] = {
    "STC1": "Identifiers must be unique and disjoint across nodes, variables, and actors.",
    "STC2": "Each node must have one valid type; branch gateways must be in {AND, OR, XOR}.",
    "STC3": "The model must declare exactly one init node.",
    "STC4": "The init node must not have incoming precedence edges.",
    "STC5": "The model must include at least one end node.",
    "STC6": "End nodes must not have outgoing precedence edges.",
    "STC7": "Action text, actor names, and variable names must be non-empty.",
    "STC8": "Each action node must have exactly one incoming and one outgoing edge.",
    "STC9": "Each branch node must have exactly one incoming and at least two outgoing edges.",
    "STC10": "Each converge node must have at least two incoming and exactly one outgoing edge.",
    "STC11": "Each variable definition must have a valid range with at least two distinct values.",
    "STC12": "Every precedence relation must connect existing nodes.",
    "STC13": "Every guard must be attached to an existing precedence edge.",
    "STC14": "Every guard condition must be a syntactically valid boolean expression.",
    "STC15": "Every variable used in a guard must be formally defined.",
    "STC16": "Every outgoing edge of XOR/OR branches must have exactly one guard.",
    "STC17": "Guards are only allowed on outgoing edges of XOR/OR branches.",
    "STC18": "Labels are only allowed on existing precedence edges.",
    "STC19": "Every non-init node must be reachable from the init node.",
    "STC20": "Every non-end node must have a path to at least one end node.",
    "STC21": "Each actor declaration must own at least one assigned node.",
    "STC22": "Every node must be assigned to exactly one actor.",
}

_LSC_DESCRIPTIONS: Dict[str, str] = {
    "LSC1": "Every guard must be satisfiable under the declared ranges of its referenced variables.",
    "LSC2": "Guards outgoing from an XOR branch must be mutually exclusive.",
    "LSC3": "Guards outgoing from XOR/OR branches must cover the whole evaluated state space.",
}


def _format_state_text(state: Any) -> str:
    if not isinstance(state, dict):
        return str(state)
    items = [f"{k}={v}" for k, v in sorted(state.items())]
    return ", ".join(items)


def _format_violation_text(check_id: str, violation: Any) -> str:
    if isinstance(violation, str):
        if violation == "missing-end-node":
            return "No end node is declared."
        return violation

    if not isinstance(violation, dict):
        return str(violation)

    issue = str(violation.get("issue", "")).strip()
    if issue == "invalid-gateway":
        return (
            f"Node `{violation.get('id')}` is a branch with invalid gateway "
            f"`{violation.get('gateway')}`."
        )
    if issue == "converge-must-not-have-gateway":
        return f"Converge node `{violation.get('id')}` incorrectly has gateway `{violation.get('gateway')}`."
    if issue == "action-content-must-be-non-empty":
        return f"Action node `{violation.get('node')}` has empty action text."
    if issue == "actor-name-must-be-non-empty":
        return f"Actor `{violation.get('actor')}` has an empty name."
    if issue == "variable-name-must-be-non-empty":
        return f"Variable `{violation.get('variable')}` has an empty name."
    if issue == "range-must-contain-at-least-two-distinct-values":
        return f"Variable `{violation.get('X')}` has an invalid range `{violation.get('R_X')}`."
    if issue == "unknown-node":
        return f"Precedence edge `{violation.get('A')} -> {violation.get('B')}` references unknown node(s)."
    if issue == "guard-without-precedence":
        return (
            f"Guard `{violation.get('condition')}` is attached to `{violation.get('A')} -> {violation.get('B')}` "
            "without a precedence edge."
        )
    if issue == "invalid-condition-expression":
        return f"Guard `{violation.get('condition')}` on `{violation.get('A')} -> {violation.get('B')}` is invalid syntax."
    if issue == "undefined-variable":
        return (
            f"Guard `{violation.get('condition')}` on `{violation.get('A')} -> {violation.get('B')}` "
            f"uses undefined variable `{violation.get('variable')}`."
        )
    if issue == "missing-or-multiple-guards-for-xor-or-or-branch":
        edge = violation.get("edge") or []
        if isinstance(edge, list) and len(edge) == 2:
            return (
                f"XOR/OR edge `{edge[0]} -> {edge[1]}` has {violation.get('guard_count')} guards; "
                "exactly one is required."
            )
        return "An XOR/OR outgoing edge has missing or multiple guards."
    if issue == "guard-does-not-originate-from-xor-or-or-branch":
        return f"Guard `{violation.get('condition')}` starts at `{violation.get('A')}`, which is not XOR/OR branch."
    if issue == "label-without-precedence":
        return f"Label is attached to non-existing edge `{violation.get('A')} -> {violation.get('B')}`."
    if issue == "actor-assigned-nodes-must-be-non-empty":
        return f"Actor `{violation.get('actor')}` has no assigned nodes."
    if issue == "node-not-assigned-to-any-actor":
        return f"Node `{violation.get('node')}` is not assigned to any actor."
    if issue == "node-assigned-to-multiple-actors":
        return f"Node `{violation.get('node')}` is assigned to multiple actors: {violation.get('actors')}"
    if issue == "actor-references-unknown-node":
        return (
            f"Actor `{violation.get('actor')}` references unknown node `{violation.get('node')}` "
            "in assigned_nodes."
        )
    if issue == "guard-unsatisfiable-under-declared-ranges":
        return (
            f"Guard `{violation.get('condition')}` on `{violation.get('A')} -> {violation.get('B')}` "
            "cannot be true under declared variable ranges."
        )
    if issue == "guard-literal-not-in-range":
        return (
            f"Guard `{violation.get('condition')}` uses `{violation.get('variable')}={violation.get('value')}`, "
            "which is outside the declared range."
        )
    if issue == "overlapping-guards-in-xor-branch":
        state = _format_state_text(violation.get("state"))
        return f"XOR branch `{violation.get('branch')}` has overlapping guards at state [{state}]."
    if issue == "state-not-covered-by-any-guard":
        state = _format_state_text(violation.get("state"))
        return f"Branch `{violation.get('branch')}` has no enabled outgoing guard at state [{state}]."
    if issue == "no-defined-variables-referenced-by-guards":
        return f"Branch `{violation.get('branch')}` guards do not reference any declared variables."
    if issue == "guard-variable-has-no-declared-range":
        variables = violation.get("variables") or []
        names = ", ".join(f"`{v}`" for v in variables) if variables else "a variable"
        if violation.get("condition") is not None:
            return (
                f"Guard `{violation.get('condition')}` on "
                f"`{violation.get('A')} -> {violation.get('B')}` cannot be evaluated: "
                f"{names} has no declared range."
            )
        return (
            f"Branch `{violation.get('branch')}` cannot be evaluated: "
            f"{names} has no declared range."
        )
    if issue == "smt-error":
        return f"SMT solver error: {violation.get('detail') or 'unknown error'}"
    if issue == "requires-exactly-one-init":
        return "This reachability check requires exactly one init node."
    if issue == "requires-at-least-one-end":
        return "This reachability check requires at least one end node."

    if check_id in {"STC8", "STC9", "STC10"} and "node" in violation:
        return (
            f"Node `{violation.get('node')}` has in={violation.get('in')} and out={violation.get('out')}."
        )

    if check_id == "STC1":
        if "E_intersect_V" in violation:
            return f"Node/variable id overlap: {violation.get('E_intersect_V')}"
        if "E_intersect_A" in violation:
            return f"Node/actor id overlap: {violation.get('E_intersect_A')}"
        if "V_intersect_A" in violation:
            return f"Variable/actor id overlap: {violation.get('V_intersect_A')}"

    return json.dumps(violation, ensure_ascii=True)


def _violation_dsl_lines(
    violation: Any,
    id_to_dsl: Dict[str, str],
    ref_to_dsl: Dict[str, str],
) -> List[str]:
    if not (id_to_dsl or ref_to_dsl):
        return []

    lines: List[str] = []
    seen: Set[str] = set()

    def push(dsl: Optional[str]) -> None:
        if dsl and dsl not in seen:
            seen.add(dsl)
            lines.append(dsl)


    if isinstance(violation, str):
        push(id_to_dsl.get(violation.strip()))
        return lines

    if not isinstance(violation, dict):
        return []


    for key in ("node", "id"):
        node_id = str(violation.get(key, "")).strip()
        if node_id:
            push(id_to_dsl.get(node_id))


    src = str(violation.get("A", "")).strip()
    dst = str(violation.get("B", "")).strip()
    if src and dst:
        edge_dsl = ref_to_dsl.get(f"guard:{src}->{dst}") or ref_to_dsl.get(f"label:{src}->{dst}")
        if edge_dsl:
            push(edge_dsl)
        elif not lines:
            push(id_to_dsl.get(src))
            push(id_to_dsl.get(dst))

    return lines


def _format_check_section(
    title: str,
    constraints: Any,
    prefix: str,
    descriptions: Dict[str, str],
    id_prefix: Optional[str] = None,
    id_to_dsl: Optional[Dict[str, str]] = None,
    ref_to_dsl: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    id_to_dsl = id_to_dsl or {}
    ref_to_dsl = ref_to_dsl or {}
    if not isinstance(constraints, dict):
        return None

    keys = sorted(
        [k for k in constraints if re.fullmatch(rf"{prefix}\d+", k)],
        key=lambda k: int(k[len(prefix):]),
    )
    if not keys:
        return None

    failed = [k for k in keys if constraints.get(k) is not True]
    if not failed:
        return None

    lines: List[str] = []
    item_index = 0
    for key in failed:
        description = descriptions.get(key, "Constraint failed.")
        violations = _as_list(constraints.get(f"{key}_violations"))
        parts: List[str] = [f"{key} failed: {description}"]
        details = constraints.get(f"{key}_details")
        if details:
            parts.append(_format_violation_text(key, details))

        if violations:
            if all(not isinstance(v, dict) for v in violations):
                parts.append(
                    "Violations: " + ", ".join(_format_violation_text(key, violation) for violation in violations)
                )
            else:
                parts.extend(_format_violation_text(key, violation) for violation in violations)


        dsl_lines: List[str] = []
        dsl_seen: Set[str] = set()
        for source in [details, *violations]:
            for dsl in _violation_dsl_lines(source, id_to_dsl, ref_to_dsl):
                if dsl not in dsl_seen:
                    dsl_seen.add(dsl)
                    dsl_lines.append(dsl)
        if dsl_lines:
            shown = dsl_lines[:8]
            suffix = f" (+{len(dsl_lines) - len(shown)} more)" if len(dsl_lines) > len(shown) else ""
            parts.append("DSL: " + "; ".join(shown) + suffix)

        item_index += 1
        tag = f"[{id_prefix}{item_index}] " if id_prefix else ""
        lines.append(f"- {tag}{' | '.join(parts)}")

    return f"{title}\n" + "\n".join(lines)


def _build_text_to_input_id(input_trace_elements: Any) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for elem in _as_list(input_trace_elements):
        if not isinstance(elem, dict):
            continue
        content = str(elem.get("content", "")).strip()
        elem_id = str(elem.get("id", "")).strip()
        if content and elem_id and content not in mapping:
            mapping[content] = elem_id
    return mapping


def _tag_text_with_input_id(text: str, text_to_input_id: Dict[str, str]) -> str:
    clean = str(text).strip()
    input_id = text_to_input_id.get(clean)
    return f"{input_id}:{clean}" if input_id else clean


def _escape_dsl_quote(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def _build_ref_to_dsl(model: Any) -> Dict[str, str]:
    model = model or {}

    elements: Dict[str, Dict[str, Any]] = {}
    for elem in _as_list(model.get("elements")):
        if isinstance(elem, dict) and elem.get("id"):
            elements[str(elem["id"])] = elem

    labels: Dict[Tuple[str, str], str] = {}
    guards: Dict[Tuple[str, str], str] = {}
    for rel in _as_list(model.get("relationships")):
        if not isinstance(rel, dict):
            continue
        rtype = str(rel.get("type", "")).strip().lower()
        src = str(rel.get("A", "")).strip()
        dst = str(rel.get("B", "")).strip()
        if not src or not dst:
            continue
        if rtype == "label":
            labels[(src, dst)] = str(rel.get("str", "")).strip()
        elif rtype == "guard":
            guards[(src, dst)] = str(rel.get("condition", "")).strip()

    def node_dsl(ref: str) -> Optional[str]:
        kind, _, node_id = ref.partition(":")
        node_id = node_id.strip()
        if not node_id:
            return None
        elem = elements.get(node_id, {})
        elem_kind = str(elem.get("type", kind)).strip().lower() or kind
        if elem_kind == "action":
            content = str(elem.get("content", "")).strip()
            return f'action({node_id},"{_escape_dsl_quote(content)}")'
        if elem_kind == "branch":
            gateway = str(elem.get("gateway", "")).strip().upper()
            return f"branch({node_id},{gateway})" if gateway else f"branch({node_id})"
        if elem_kind in ("init", "end", "converge"):
            return f"{elem_kind}({node_id})"
        return f"{elem_kind}({node_id})"

    ref_to_dsl: Dict[str, str] = {}
    for ref in (f"{e.get('type','')}:{eid}" for eid, e in elements.items()):
        dsl = node_dsl(ref)
        if dsl:
            ref_to_dsl[ref] = dsl

    for (src, dst), text in labels.items():
        ref_to_dsl[f"label:{src}->{dst}"] = f'label({src},{dst},"{_escape_dsl_quote(text)}")'
    for (src, dst), cond in guards.items():
        ref_to_dsl[f"guard:{src}->{dst}"] = f'guard({src},{dst},"{_escape_dsl_quote(cond)}")'

    for var in _as_list(model.get("variable_definitions")):
        if not isinstance(var, dict):
            continue
        var_id = str(var.get("X", "")).strip()
        name = str(var.get("name", "")).strip()
        rng = var.get("R_X")
        values = ", ".join(str(v).strip() for v in rng) if isinstance(rng, list) else ""
        key_id = var_id or name
        if key_id:
            ref_to_dsl[f"variable:{key_id}"] = (
                f'defines({var_id},"{_escape_dsl_quote(name)}",{{{values}}})'
            )

    for actor in _as_list(model.get("actors")):
        if not isinstance(actor, dict):
            continue
        actor_id = str(actor.get("id", "")).strip()
        name = str(actor.get("name", "")).strip()
        assigned = actor.get("assigned_nodes") or []
        assigned_list = ", ".join(str(a).strip() for a in assigned) if isinstance(assigned, list) else ""
        key_id = actor_id or name
        if key_id:
            ref_to_dsl[f"actor:{key_id}"] = (
                f'actor({actor_id},"{_escape_dsl_quote(name)}",[{assigned_list}])'
            )

    return ref_to_dsl


def _refs_to_dsl(refs: Any, ref_to_dsl: Dict[str, str]) -> str:
    rendered: List[str] = []
    for ref in _as_list(refs):
        clean = str(ref).strip()
        if not clean:
            continue
        rendered.append(ref_to_dsl.get(clean, clean))
    return " -> ".join(rendered)


def _build_id_to_dsl(ref_to_dsl: Dict[str, str]) -> Dict[str, str]:
    id_to_dsl: Dict[str, str] = {}
    for ref, dsl in ref_to_dsl.items():
        kind, _, ident = ref.partition(":")
        if not ident or "->" in ident:
            continue
        id_to_dsl[ident] = dsl
    return id_to_dsl


def _render_model_trace(trace: Any, id_to_dsl: Dict[str, str]) -> str:
    if not isinstance(trace, dict):
        return _join_path_text(trace)
    ids = _as_list(trace.get("ids"))
    contents = _as_list(trace.get("content"))
    rendered: List[str] = []
    for idx, ident in enumerate(ids):
        ident_s = str(ident).strip()
        if not ident_s:
            continue
        dsl = id_to_dsl.get(ident_s)
        if dsl:
            rendered.append(dsl)
        else:
            text = str(contents[idx]).strip() if idx < len(contents) else ""
            rendered.append(f"{ident_s}:{text}" if text else ident_s)
    if rendered:
        return " -> ".join(rendered)
    return _join_path_text(contents)


def _render_input_trace(trace: Any) -> str:
    if not isinstance(trace, dict):
        return _join_path_text(trace)
    ids = _as_list(trace.get("ids"))
    contents = _as_list(trace.get("content"))
    rendered: List[str] = []
    for idx, ident in enumerate(ids):
        ident_s = str(ident).strip()
        text = str(contents[idx]).strip() if idx < len(contents) else ""
        if ident_s and text:
            rendered.append(f"{ident_s}:{text}")
        elif text:
            rendered.append(text)
        elif ident_s:
            rendered.append(ident_s)
    if rendered:
        return " -> ".join(rendered)
    return _join_path_text(contents)


def format_consolidated_feedback(
    mapping_result: Any,
    evaluation_result: Any,
    var_actor_result: Any = None,
    constraints: Any = None,
    model: Any = None,
    input_trace_elements: Any = None,
) -> str:
    text_to_input_id = _build_text_to_input_id(input_trace_elements)
    ref_to_dsl = _build_ref_to_dsl(model if isinstance(model, dict) else {})
    id_to_dsl = _build_id_to_dsl(ref_to_dsl)
    mapping_dict = mapping_result.to_dict() if hasattr(mapping_result, "to_dict") else mapping_result
    evaluation_dict = evaluation_result.to_dict() if hasattr(evaluation_result, "to_dict") else evaluation_result
    mapping_dict = mapping_dict if isinstance(mapping_dict, dict) else {}
    evaluation_dict = evaluation_dict if isinstance(evaluation_dict, dict) else {}

    var_actor_dict: Dict[str, Any] = {}
    if var_actor_result is not None:
        var_actor_dict = var_actor_result.to_dict() if hasattr(var_actor_result, "to_dict") else var_actor_result
        var_actor_dict = var_actor_dict if isinstance(var_actor_dict, dict) else {}

    omissions: List[str] = []
    additions: List[str] = []
    mutations: List[str] = []
    permutations: List[str] = []
    seen: Dict[int, Set[str]] = {id(b): set() for b in (omissions, additions, mutations, permutations)}

    def add_unique(bucket: List[str], line: str) -> None:
        clean = str(line).strip()
        if not clean:
            return
        key = _dedup_key(clean)
        bucket_seen = seen[id(bucket)]
        if key in bucket_seen:
            return
        bucket_seen.add(key)
        bucket.append(clean)


    for omission in _as_list(mapping_dict.get("omissions")):
        if not isinstance(omission, dict):
            continue
        reason = str(omission.get("reason", "")).strip()
        for text in _as_list(omission.get("verbatim_texts")):
            tagged = _tag_text_with_input_id(text, text_to_input_id)
            entry = f"Node-level omission: {tagged}"
            if reason:
                entry += f" (reason: {reason})"
            add_unique(omissions, entry)

    for addition in _as_list(mapping_dict.get("additions")):
        if not isinstance(addition, dict):
            continue
        dsl = _refs_to_dsl(addition.get("refs"), ref_to_dsl)
        reason = str(addition.get("reason", "")).strip()
        entry = f"Node-level addition: {dsl}" if dsl else "Node-level addition"
        if reason:
            entry += f" (reason: {reason})"
        add_unique(additions, entry)

    for mutation in _as_list(mapping_dict.get("mutations")):
        if not isinstance(mutation, dict):
            continue
        parts = []
        if mutation.get("refs"):
            parts.append(f"element={_refs_to_dsl(mutation.get('refs'), ref_to_dsl)}")
        if mutation.get("verbatim_texts"):
            tagged = " -> ".join(
                _tag_text_with_input_id(t, text_to_input_id)
                for t in _as_list(mutation.get("verbatim_texts"))
                if str(t).strip()
            )
            parts.append(f"source={tagged}")
        if str(mutation.get("reason", "")).strip():
            parts.append(f"reason={str(mutation.get('reason')).strip()}")
        add_unique(mutations, f"Node-level mutation: {'; '.join(parts) or 'details unavailable'}")


    for ce in _as_list(evaluation_dict.get("counter_examples")):
        if not isinstance(ce, dict):
            continue
        ce_type = str(ce.get("type", "")).strip().lower()
        if ce_type == "missing_trace":
            text = _render_input_trace(ce.get("expected_trace", {}))
            add_unique(omissions, f"Omitted trace (input {ce.get('input_trace_idx')}): {text}")
        elif ce_type == "hallucinated_trace":
            text = _render_model_trace(ce.get("actual_trace", {}), id_to_dsl)
            add_unique(additions, f"Added trace (model {ce.get('gen_trace_idx')}): {text}")


    for match in _as_list(evaluation_dict.get("trace_matches")):
        if not isinstance(match, dict):
            continue
        in_idx = match.get("input_trace_idx")
        gen_idx = match.get("gen_trace_idx")
        for div in _as_list(match.get("divergences")):
            if not isinstance(div, dict):
                continue
            div_type = str(div.get("type", "")).strip().lower()
            if div_type == "omission":
                text = _render_input_trace(div.get("expected", {}))
                add_unique(omissions, f"Within matched traces ({in_idx}->{gen_idx}) omitted: {text}")
            elif div_type == "addition":
                text = _render_model_trace(div.get("actual", {}), id_to_dsl)
                add_unique(additions, f"Within matched traces ({in_idx}->{gen_idx}) added: {text}")
            elif div_type == "reordering":
                exp = _render_input_trace(div.get("expected", {}))
                act = _render_model_trace(div.get("actual", {}), id_to_dsl)
                line = (
                    f"Within matched traces ({in_idx}->{gen_idx}) reordered: "
                    f"expected [{exp}] but observed [{act}] elsewhere"
                )
                swapped = _render_model_trace(div.get("swapped_with", {}), id_to_dsl)
                if swapped:
                    line += f"; in its place the model has [{swapped}]"
                add_unique(permutations, line)


    def ingest_smc(scope: str, bucket: Any) -> None:
        if not isinstance(bucket, dict):
            return
        for mutation in _as_list(bucket.get("mutations")):
            if not isinstance(mutation, dict):
                continue
            parts = []
            if mutation.get("refs"):
                parts.append(f"ref={_refs_to_dsl(mutation.get('refs'), ref_to_dsl)}")
            if str(mutation.get("reason", "")).strip():
                parts.append(f"reason={str(mutation.get('reason')).strip()}")
            add_unique(mutations, f"{scope} mutation: {'; '.join(parts) or 'details unavailable'}")
        for omission in _as_list(bucket.get("omissions")):
            if not isinstance(omission, dict):
                continue
            entry = f"{scope} omission"
            if str(omission.get("summary", "")).strip():
                entry += f": {str(omission.get('summary')).strip()}"
            if str(omission.get("reason", "")).strip():
                entry += f" (reason: {str(omission.get('reason')).strip()})"
            add_unique(omissions, entry)
        for addition in _as_list(bucket.get("additions")):
            if not isinstance(addition, dict):
                continue
            entry = f"{scope} addition"
            dsl = _refs_to_dsl(addition.get("refs"), ref_to_dsl)
            if dsl:
                entry += f": {dsl}"
            if str(addition.get("reason", "")).strip():
                entry += f" (reason: {str(addition.get('reason')).strip()})"
            add_unique(additions, entry)

    if var_actor_dict:
        variables_bucket = var_actor_dict.get("SMC6")
        actors_bucket = var_actor_dict.get("SMC7")
        ingest_smc("Variable-level", variables_bucket)
        ingest_smc("Actor-level", actors_bucket)


    neuro_counter = {"n": 0}

    def render(title: str, lines: List[str]) -> Optional[str]:
        if not lines:
            return None
        rendered = []
        for line in lines:
            neuro_counter["n"] += 1
            rendered.append(f"- [N{neuro_counter['n']}] {line}")
        return f"{title}\n" + "\n".join(rendered)

    omission_section = render("Omission", omissions)
    addition_section = render("Addition", additions)
    mutation_section = render("Mutation", mutations)
    permutation_section = render("Permutation", permutations)

    structural_section = _format_check_section(
        title="Structural Check",
        constraints=constraints,
        prefix="STC",
        descriptions=_STC_DESCRIPTIONS,
        id_prefix="S",
        id_to_dsl=id_to_dsl,
        ref_to_dsl=ref_to_dsl,
    )
    logical_section = _format_check_section(
        title="Logical Sanity Check",
        constraints=constraints,
        prefix="LSC",
        descriptions=_LSC_DESCRIPTIONS,
        id_prefix="L",
        id_to_dsl=id_to_dsl,
        ref_to_dsl=ref_to_dsl,
    )

    sections = [
        omission_section,
        addition_section,
        mutation_section,
        permutation_section,
        structural_section,
        logical_section,
    ]
    body = "\n\n".join(section for section in sections if section)
    return (body + "\n") if body else ""


def format_structural_logical_feedback(constraints: Dict[str, Any],
                                       model: Any) -> str:
    if not constraints:
        return ""
    ref_to_dsl = _build_ref_to_dsl(model if isinstance(model, dict) else {})
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
    return "\n\n".join(s for s in (structural, logical) if s).strip()


def _conformance_ratio(constraints: Dict[str, Any], prefix: str) -> Tuple[int, int]:
    passed = total = 0
    for key, value in constraints.items():

        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        if not suffix.isdigit():
            continue
        total += 1
        if value is True:
            passed += 1
    return passed, total


def _bucket_counts(bucket: Any) -> Tuple[int, int, int, int]:
    if not isinstance(bucket, dict):
        return 0, 0, 0, 0
    return (
        len(_as_list(bucket.get("matches"))),
        len(_as_list(bucket.get("mutations"))),
        len(_as_list(bucket.get("omissions"))),
        len(_as_list(bucket.get("additions"))),
    )


def _var_actor_components(var_actor_result: Any) -> Dict[str, Any]:
    smc6 = smc7 = None
    if var_actor_result is not None:
        smc6 = getattr(var_actor_result, "smc6", None)
        smc7 = getattr(var_actor_result, "smc7", None)
        if smc6 is None and smc7 is None:
            va_dict = var_actor_result.to_dict() if hasattr(var_actor_result, "to_dict") else var_actor_result
            if isinstance(va_dict, dict):
                smc6 = va_dict.get("SMC6")
                smc7 = va_dict.get("SMC7")

    v_match, v_mut, v_omit, v_add = _bucket_counts(smc6)
    a_match, a_mut, a_omit, a_add = _bucket_counts(smc7)

    matches = v_match + a_match
    mutations = v_mut + a_mut
    omissions = v_omit + a_omit
    additions = v_add + a_add

    return {
        "credit": matches + 0.5 * mutations,
        "recall_denom": matches + mutations + omissions,
        "precision_denom": matches + mutations + additions,
        "matches": matches,
        "mutations": mutations,
        "omissions": omissions,
        "additions": additions,
        "per_check": {
            "SMC6_variables": {
                "matches": v_match,
                "mutations": v_mut,
                "omissions": v_omit,
                "additions": v_add,
            },
            "SMC7_actors": {
                "matches": a_match,
                "mutations": a_mut,
                "omissions": a_omit,
                "additions": a_add,
            },
        },
    }


def compute_metrics(
    evaluation_result: Any,
    mapping_result: Any = None,
    constraints: Any = None,
    var_actor_result: Any = None,
) -> Dict[str, Any]:


    if hasattr(evaluation_result, "detailed_metrics"):
        metrics_block = evaluation_result.detailed_metrics or {}
    elif isinstance(evaluation_result, dict):
        metrics_block = evaluation_result.get("detailed_metrics", {}) or {}
    else:
        metrics_block = {}
    total_source_steps = metrics_block.get("total_input_steps", 0)
    total_model_steps = metrics_block.get("total_gen_steps", 0)


    source_credit = float(metrics_block.get("trace_source_credit", 0.0))
    model_credit = float(metrics_block.get("trace_model_credit", 0.0))


    forgiven_scaffolding = int(metrics_block.get("forgiven_scaffolding_steps", 0))
    model_credit += forgiven_scaffolding


    va = _var_actor_components(var_actor_result)
    source_credit += va["credit"]
    model_credit += va["credit"]
    total_source_steps += va["recall_denom"]
    total_model_steps += va["precision_denom"]

    recall = source_credit / total_source_steps if total_source_steps else 0.0
    precision = model_credit / total_model_steps if total_model_steps else 0.0
    recall = min(recall, 1.0)
    precision = min(precision, 1.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    constraints = constraints if isinstance(constraints, dict) else {}
    stc_passed, stc_total = _conformance_ratio(constraints, "STC")
    lsc_passed, lsc_total = _conformance_ratio(constraints, "LSC")


    all_stc_passed = bool(constraints.get("all_stc_passed", stc_passed == stc_total))
    all_lsc_passed = bool(constraints.get("all_lsc_passed", lsc_passed == lsc_total))
    constraints_ok = all_stc_passed and all_lsc_passed
    if not constraints_ok:
        recall = 0.0
        precision = 0.0
        f1 = 0.0

    def ratio(passed: int, total: int) -> float:
        return passed / total if total else 1.0

    return {
        "precision_recall": {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "totals": {
                "source_units": total_source_steps,
                "model_units": total_model_steps,
                "source_credit": source_credit,
                "model_credit": model_credit,
            },
        },
        "structural_conformance": {
            "passed": stc_passed,
            "total": stc_total,
            "ratio": ratio(stc_passed, stc_total),
        },
        "logical_sanity_conformance": {
            "passed": lsc_passed,
            "total": lsc_total,
            "ratio": ratio(lsc_passed, lsc_total),
        },
    }


def format_metrics(metrics: Dict[str, Any]) -> str:
    pr = metrics.get("precision_recall", {})
    sc = metrics.get("structural_conformance", {})
    sm = metrics.get("logical_sanity_conformance", {})
    totals = pr.get("totals", {})

    recall = pr.get("recall", pr.get("completeness_recall", 0.0)) or 0.0
    precision = pr.get("precision", pr.get("correctness_precision", 0.0)) or 0.0

    lines = ["=== METRICS ===", "", "Precision / recall (model vs. process description):"]
    lines.append(f"  Completeness (recall):   {recall:.1%}")
    lines.append(f"  Correctness (precision): {precision:.1%}")
    lines.append(f"  F1:                      {pr.get('f1', 0.0):.1%}")
    lines.append(
        f"  Pooled units: {totals.get('source_credit', 0.0):.1f}/{totals.get('source_units', 0)} source credit, "
        f"{totals.get('model_credit', 0.0):.1f}/{totals.get('model_units', 0)} model credit"
    )
    lines.append("")
    lines.append("Conformance to constraints:")
    lines.append(
        f"  Structural (STC): {sc.get('passed', 0)}/{sc.get('total', 0)} "
        f"({sc.get('ratio', 0.0):.1%})"
    )
    lines.append(
        f"  Logical sanity (LSC): {sm.get('passed', 0)}/{sm.get('total', 0)} "
        f"({sm.get('ratio', 0.0):.1%})"
    )
    if (sc.get("ratio", 1.0) or 0.0) < 1.0 or (sm.get("ratio", 1.0) or 0.0) < 1.0:
        lines.append("  -> P/R/F1 zeroed: the model violates a structural or "
                     "logical-sanity constraint")
    return "\n".join(lines) + "\n"
