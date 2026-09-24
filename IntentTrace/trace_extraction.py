from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Set, Tuple

from . import prompts
from . import parsed_model as pm


async def build_traces_holistic(
    llm_caller,
    elements: List[Dict[str, Any]],
    original_text: str,
) -> Dict[str, Any]:
    if not elements:
        return {"traces": []}

    id_list = [str(elem.get("id", str(i))) for i, elem in enumerate(elements)]
    catalog_lines = []
    for i, elem in enumerate(elements):
        elem_id = str(elem.get("id", str(i)))
        content = elem.get("main_content", elem.get("content", ""))
        catalog_lines.append(f"[ID: {elem_id}] {content}")
    element_catalog = "\n".join(catalog_lines)

    prompt = prompts.TRACE_EXTRACTION_PROMPT
    prompt = prompt.replace("{{original_text}}", original_text or "")
    prompt = prompt.replace("{{atomic_elements}}", element_catalog)

    result = await llm_caller.call_async(
        [{"role": "user", "content": prompt}],
        label="trace_builder",
    )
    if result.error:
        return {"traces": []}

    return {"traces": _parse_holistic_response(result.response or "", id_list)}


def get_all_model_traces(model: Any, max_paths: int = 10000) -> Dict[str, Any]:
    if max_paths <= 0:
        max_paths = 1

    if isinstance(model, pm.ParsedModel):
        parsed = model
    elif isinstance(model, dict):
        parsed = pm.build_parsed_model(model)
    else:
        return {"traces": []}

    node_ids, node_types, adjacency = pm.trace_graph(parsed)
    if not node_ids:
        return {"traces": []}

    start_types = {"init", "action-initial", "action_initial"}
    end_types = {"end", "action-end", "action_end"}
    start_nodes = [n for n in node_ids if node_types.get(n) in start_types]
    end_nodes = [n for n in node_ids if node_types.get(n) in end_types]

    if not start_nodes:
        incoming_counts = {n: 0 for n in node_ids}
        for children in adjacency.values():
            for child in children:
                incoming_counts[child] = incoming_counts.get(child, 0) + 1
        start_nodes = [n for n in node_ids if incoming_counts.get(n, 0) == 0]

    if not end_nodes:
        end_nodes = [n for n, children in adjacency.items() if not children]

    end_set = set(end_nodes)
    traces: List[List[str]] = []
    trace_set: Set[Tuple[str, ...]] = set()
    truncated = False

    def _dfs(current: str, path: List[str], visited: Set[str]) -> None:
        nonlocal truncated
        if truncated:
            return

        if current in end_set:
            trace_tuple = tuple(path)
            if trace_tuple not in trace_set:
                trace_set.add(trace_tuple)
                traces.append(path.copy())
            return

        for nxt in adjacency.get(current, []):
            if len(traces) >= max_paths:
                truncated = True
                return
            if nxt in visited:
                continue
            visited.add(nxt)
            path.append(nxt)
            _dfs(nxt, path, visited)
            path.pop()
            visited.remove(nxt)
            if truncated:
                return

    for start in start_nodes:
        if len(traces) >= max_paths:
            break
        _dfs(start, [start], {start})
        if truncated:
            break

    return {"traces": traces}


def _parse_holistic_response(response: str, valid_ids: List[str]) -> List[List[str]]:
    try:
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response, re.DOTALL)
        clean_json = match.group(1) if match else response
        data = json.loads(clean_json.strip())
        if not isinstance(data, list):
            return []

        valid_set = set(valid_ids)
        sanitized: List[List[str]] = []
        for trace in data:
            if not isinstance(trace, list):
                continue
            kept = [str(node_id) for node_id in trace if str(node_id) in valid_set]
            if kept:
                sanitized.append(kept)
        return sanitized
    except Exception:
        return []
