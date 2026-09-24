from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from . import prompts
from . import parsed_model as pm


class MappingResult:

    def __init__(
        self,
        matches: List[Dict[str, Any]],
        mutations: List[Dict[str, Any]],
        omissions: List[Dict[str, Any]],
        additions: List[Dict[str, Any]],
        targets: Optional[List[Dict[str, Any]]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        raw_response: Optional[str] = None,
        model_id: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        spec_elements: Optional[List[Dict[str, Any]]] = None,
        spec_traces: Optional[List[List[str]]] = None,
        spec_revisions: Optional[List[Dict[str, Any]]] = None,
    ):
        self.matches = matches
        self.mutations = mutations
        self.omissions = omissions
        self.additions = additions
        self.targets = targets or []
        self.metrics = metrics or {}
        self.raw_response = raw_response
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.spec_elements = spec_elements
        self.spec_traces = spec_traces
        self.spec_revisions = spec_revisions

    @property
    def revised_spec(self) -> bool:
        return self.spec_elements is not None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "matches": self.matches,
            "mutations": self.mutations,
            "omissions": self.omissions,
            "additions": self.additions,
            "targets": self.targets,
            "metrics": self.metrics,
            "model_id": self.model_id,
            "reasoning_effort": self.reasoning_effort,
        }
        if self.revised_spec:
            payload["spec_elements"] = self.spec_elements
            payload["spec_traces"] = self.spec_traces or []
            payload["spec_revisions"] = self.spec_revisions or []
        return payload


async def match_elements(
    input_text: str,
    model: Any,
    llm_caller,
    model_id: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    prompt_template: Optional[str] = None,
    input_elements: Optional[List[Dict[str, Any]]] = None,
    spec_elements: Optional[List[Dict[str, Any]]] = None,
    spec_traces: Optional[List[List[str]]] = None,
) -> MappingResult:
    parsed = _coerce_parsed_model(model)
    targets = _build_targets(parsed)
    elements_payload = _build_elements_prompt_payload(targets)

    given_spec = _clean_spec_elements(spec_elements)
    use_spec = given_spec is not None

    input_elements_payload = ""
    for elem in input_elements or []:
        eid = str(elem.get("id", "")).strip()
        content = str(elem.get("content", "")).strip()
        if eid and content:
            input_elements_payload += f"{eid}: {content}\n"
    input_elements_payload = input_elements_payload.strip()

    if prompt_template is not None:
        prompt = prompt_template
    elif use_spec:
        prompt = prompts.ELEMENT_MATCHING_SPEC_PROMPT
    else:
        prompt = prompts.ELEMENT_MATCHING_PROMPT
    prompt = prompt.replace("{{input_text}}", input_text or "")
    prompt = prompt.replace("{{elements}}", elements_payload)
    prompt = prompt.replace("{{input_elements}}", input_elements_payload)
    prompt = prompt.replace("{{spec_elements}}", _render_spec_elements(given_spec))
    prompt = prompt.replace("{{spec_traces}}", _render_spec_traces(spec_traces))

    def _fallback_spec() -> Dict[str, Any]:
        if not use_spec:
            return {}
        return {
            "spec_elements": given_spec,
            "spec_traces": _clean_spec_traces(spec_traces, given_spec),
            "spec_revisions": [],
        }

    result = await llm_caller.call_async(
        [{"role": "user", "content": prompt}],
        label="element_mapping",
    )
    if result.error:
        auto_additions = _build_uncovered_additions([], [], [], targets)
        return MappingResult(
            matches=[],
            mutations=[],
            omissions=[],
            additions=auto_additions,
            targets=targets,
            metrics=_calculate_metrics([], [], [], auto_additions, input_text, parsed, targets),
            raw_response=str(result.error),
            model_id=model_id,
            reasoning_effort=reasoning_effort,
            **_fallback_spec(),
        )

    response = result.response
    raw_data = _parse_matching_response(response) or {}
    normalized = _normalize_mapping_data(raw_data, targets)

    matches = normalized["matches"]
    mutations = normalized["mutations"]
    omissions = normalized["omissions"]
    additions = normalized["additions"]
    additions.extend(_build_uncovered_additions(matches, mutations, additions, targets))

    spec_payload = _fallback_spec()
    if use_spec:
        revised = _normalize_spec_revision(raw_data, given_spec, spec_traces)
        if revised is not None:
            spec_payload = revised

    metrics = _calculate_metrics(
        matches=matches,
        mutations=mutations,
        omissions=omissions,
        additions=additions,
        input_text=input_text,
        elements=parsed,
        targets=targets,
    )

    return MappingResult(
        matches=matches,
        mutations=mutations,
        omissions=omissions,
        additions=additions,
        targets=targets,
        metrics=metrics,
        raw_response=response,
        model_id=model_id,
        reasoning_effort=reasoning_effort,
        **spec_payload,
    )


def _coerce_parsed_model(model: Any) -> pm.ParsedModel:
    if isinstance(model, pm.ParsedModel):
        return model
    if isinstance(model, dict):
        return pm.build_parsed_model(model)
    return pm.build_parsed_model({})


def _is_content_target(target: Dict[str, Any]) -> bool:
    return pm.is_content_kind(target.get("kind"))


def _render_source_line(target: Dict[str, Any]) -> str:
    kind = target.get("kind")
    nid = target.get("node_id")
    content = target.get("content", "")
    frm = target.get("from")
    to = target.get("to")
    if kind == "action":
        return f"action {nid}: {content}"
    if kind == "branch":
        gateway = target.get("gateway", "")
        return f"branch {nid} ({gateway})" if gateway else f"branch {nid}"
    if kind in ("init", "end", "converge"):
        return f"{kind} {nid}"
    if kind == "label":
        return f"label {frm}->{to}: {content}"
    if kind == "guard":
        return f"guard {frm}->{to}: {content}"
    return str(target.get("ref", ""))


def _build_targets(parsed: pm.ParsedModel) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    for t in pm.mapping_targets(parsed):
        ref = t["ref"]
        aliases = [ref]
        if t.get("node_id"):
            aliases.append(str(t["node_id"]))
        target: Dict[str, Any] = {
            "ref_id": ref,
            "kind": t["kind"],
            "node_id": t.get("node_id"),
            "gateway": t.get("gateway", ""),
            "string_value": t.get("content", "") or t["kind"],
            "aliases": aliases,
            "from": t.get("from"),
            "to": t.get("to"),
        }
        target["source_line"] = _render_source_line(t)
        if t["kind"] == "guard":
            target["guard"] = {
                "from": t.get("from"),
                "to": t.get("to"),
                "condition": t.get("content", ""),
            }
        targets.append(target)
    return targets


def _build_elements_prompt_payload(targets: List[Dict[str, Any]]) -> str:
    elements: List[Dict[str, Any]] = []
    for t in targets:
        entry: Dict[str, Any] = {
            "ref": t["ref_id"],
            "kind": t["kind"],
        }
        if t["kind"] == "branch" and t.get("gateway"):
            entry["gateway"] = t["gateway"]
        if t["kind"] in ("action", "label", "guard"):
            entry["content"] = t.get("string_value", "")
        if t["kind"] in ("label", "guard"):
            entry["from"] = t.get("from")
            entry["to"] = t.get("to")
        elements.append(entry)
    return json.dumps(elements, indent=2)


def _parse_matching_response(response: str) -> Optional[Dict[str, Any]]:
    cleaned = (response or "").strip()
    if not cleaned:
        return None

    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()

    parsed_dict: Optional[Dict[str, Any]] = None
    try:
        loaded = json.loads(cleaned)
        if isinstance(loaded, dict):
            parsed_dict = loaded
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                loaded = json.loads(cleaned[start:end])
                if isinstance(loaded, dict):
                    parsed_dict = loaded
            except json.JSONDecodeError:
                parsed_dict = None

    if not parsed_dict:
        return None

    return {
        "matches": _ensure_dict_list(parsed_dict.get("matches")),
        "mutations": _ensure_dict_list(parsed_dict.get("mutations")),
        "omissions": _ensure_dict_list(parsed_dict.get("omissions")),
        "additions": _ensure_dict_list(parsed_dict.get("additions")),
        "spec_elements": _ensure_dict_list(parsed_dict.get("spec_elements")),
        "spec_traces": parsed_dict.get("spec_traces"),
        "spec_revisions": _ensure_dict_list(parsed_dict.get("spec_revisions")),
    }


def _ensure_dict_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]



def _clean_spec_elements(elements: Any) -> Optional[List[Dict[str, str]]]:
    if elements is None:
        return None
    if not isinstance(elements, list):
        return None

    cleaned: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for elem in elements:
        if not isinstance(elem, dict):
            continue
        eid = str(elem.get("id", "")).strip()
        content = str(elem.get("content", "")).strip()
        if not eid or not content or eid in seen:
            continue
        seen.add(eid)
        cleaned.append({"id": eid, "content": content})
    return cleaned


def _clean_spec_traces(traces: Any, elements: Optional[List[Dict[str, str]]]) -> List[List[str]]:
    if not isinstance(traces, list):
        return []
    valid = {e["id"] for e in elements or []}
    out: List[List[str]] = []
    for trace in traces:
        if not isinstance(trace, list):
            continue
        steps = [str(s).strip() for s in trace if str(s).strip()]
        steps = [s for s in steps if s in valid] if valid else steps
        if steps:
            out.append(steps)
    return out


def _render_spec_elements(elements: Optional[List[Dict[str, str]]]) -> str:
    if not elements:
        return "None"
    return "\n".join(f"{e['id']}: {e['content']}" for e in elements)


def _render_spec_traces(traces: Any) -> str:
    cleaned = _clean_spec_traces(traces, None)
    if not cleaned:
        return "None"
    return "\n".join(
        f"Trace {n}: {' -> '.join(trace)}" for n, trace in enumerate(cleaned, 1)
    )


def _next_spec_id(existing: Set[str]) -> str:
    highest = 0
    for eid in existing:
        match = re.fullmatch(r"i(\d+)", eid)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"i{highest + 1}"


def _normalize_spec_revision(
    raw_data: Dict[str, Any],
    given: List[Dict[str, str]],
    given_traces: Any,
) -> Optional[Dict[str, Any]]:
    returned = _clean_spec_elements(raw_data.get("spec_elements"))
    if not returned:
        return None

    given_by_id = {e["id"]: e["content"] for e in given}
    used_ids: Set[str] = set(given_by_id)

    elements: List[Dict[str, str]] = []
    seen: Set[str] = set()
    renamed: Dict[str, str] = {}
    for elem in returned:
        eid, content = elem["id"], elem["content"]
        if eid in seen:
            continue
        if eid not in given_by_id and re.fullmatch(r"i\d+", eid) is None:
            new_id = _next_spec_id(used_ids)
            renamed[eid] = new_id
            eid = new_id
        used_ids.add(eid)
        seen.add(eid)
        elements.append({"id": eid, "content": content})

    traces = raw_data.get("spec_traces")
    if not isinstance(traces, list):
        traces = given_traces
    if renamed:
        traces = [
            [renamed.get(str(s).strip(), str(s).strip()) for s in trace]
            for trace in traces or []
            if isinstance(trace, list)
        ]
    traces = _clean_spec_traces(traces, elements)

    revisions: List[Dict[str, Any]] = []
    for entry in _ensure_dict_list(raw_data.get("spec_revisions")):
        action = str(entry.get("action", "")).strip().lower()
        ids = [str(i).strip() for i in entry.get("ids", []) or [] if str(i).strip()]
        ids = [renamed.get(i, i) for i in ids]
        if action not in ("kept", "edited", "added", "removed") or not ids:
            continue
        revisions.append({
            "action": action,
            "ids": ids,
            "reason": str(entry.get("reason", "")).strip(),
        })

    returned_ids = {e["id"] for e in elements}
    derived: List[Dict[str, Any]] = []
    for elem in elements:
        eid, content = elem["id"], elem["content"]
        if eid not in given_by_id:
            derived.append({"action": "added", "ids": [eid]})
        elif given_by_id[eid] != content:
            derived.append({"action": "edited", "ids": [eid]})
    for eid in given_by_id:
        if eid not in returned_ids:
            derived.append({"action": "removed", "ids": [eid]})

    reason_for: Dict[Tuple[str, str], str] = {
        (r["action"], i): r["reason"] for r in revisions for i in r["ids"]
    }
    for entry in derived:
        entry["reason"] = reason_for.get((entry["action"], entry["ids"][0]), "")

    return {
        "spec_elements": elements,
        "spec_traces": traces,
        "spec_revisions": derived,
    }


def _normalize_mapping_data(
    raw_data: Dict[str, Any],
    targets: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    lookup = _build_target_lookup(targets)
    target_map = _resolve_target_map(targets)

    matches = [_normalize_match_entry(entry, lookup) for entry in raw_data.get("matches", [])]
    matches = _sanitize_relation_entries(matches, target_map, reason_key="justification")

    mutations = [_normalize_mutation_entry(entry, lookup) for entry in raw_data.get("mutations", [])]
    mutations = _sanitize_relation_entries(mutations, target_map, reason_key="reason")

    omissions = [_normalize_omission_entry(entry) for entry in raw_data.get("omissions", [])]
    omissions = [entry for entry in omissions if entry.get("verbatim_texts")]

    additions = [_normalize_addition_entry(entry, lookup, target_map) for entry in raw_data.get("additions", [])]
    additions = [entry for entry in additions if entry.get("refs")]
    additions = _split_multi_ref_entries(additions)

    return {
        "matches": matches,
        "mutations": mutations,
        "omissions": omissions,
        "additions": additions,
    }


def _sanitize_relation_entries(
    entries: List[Dict[str, Any]],
    target_map: Dict[str, Dict[str, Any]],
    reason_key: str,
) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []

    for entry in entries:
        refs = [ref for ref in entry.get("refs", []) if ref in target_map]
        texts = [text for text in entry.get("verbatim_texts", []) if str(text).strip()]

        if not refs or not texts:
            continue

        content_refs = [
            ref for ref in refs if _is_content_target(target_map.get(ref, {}))
        ]
        structural_refs = [ref for ref in refs if ref not in content_refs]

        if len(refs) > 1 and structural_refs:
            refs = content_refs

        if not refs:
            continue

        if len(refs) > 1 and len(texts) > 1:
            rationale = str(entry.get(reason_key, "")).strip()

            if len(refs) == len(texts):
                for ref, text in zip(refs, texts):
                    sanitized.append({
                        "refs": [ref],
                        "verbatim_texts": [text],
                        reason_key: rationale,
                    })
                continue

            for ref in refs:
                sanitized.append({
                    "refs": [ref],
                    "verbatim_texts": texts,
                    reason_key: rationale,
                })
            continue

        normalized = dict(entry)
        normalized["refs"] = refs
        normalized["verbatim_texts"] = texts
        sanitized.append(normalized)

    return sanitized


def _build_target_lookup(targets: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    lookup: Dict[str, Set[str]] = {}

    def add(key: Any, ref_id: str) -> None:
        if key is None:
            return
        k = str(key).strip()
        if not k:
            return
        lookup.setdefault(k, set()).add(ref_id)

    for target in targets:
        ref_id = str(target.get("ref_id", "")).strip()
        if not ref_id:
            continue
        add(ref_id, ref_id)
        add(target.get("node_id"), ref_id)
        add(target.get("source_line"), ref_id)
        for alias in target.get("aliases", []):
            add(alias, ref_id)

    return lookup


def _extract_raw_refs(entry: Dict[str, Any]) -> List[str]:
    refs: List[str] = []

    list_value = entry.get("refs")
    if isinstance(list_value, list):
        refs.extend(str(item).strip() for item in list_value if str(item).strip())
    elif isinstance(list_value, str) and list_value.strip():
        refs.append(list_value.strip())

    for key in ("ref", "node_id", "ref_id"):
        value = entry.get(key)
        if value is None:
            continue
        as_str = str(value).strip()
        if as_str:
            refs.append(as_str)

    seen: Set[str] = set()
    unique_refs: List[str] = []
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        unique_refs.append(ref)
    return unique_refs


def _resolve_refs(raw_refs: List[str], lookup: Dict[str, Set[str]]) -> List[str]:
    resolved: List[str] = []
    seen: Set[str] = set()
    for raw_ref in raw_refs:
        for ref_id in sorted(lookup.get(raw_ref, set())):
            if ref_id in seen:
                continue
            seen.add(ref_id)
            resolved.append(ref_id)
    return resolved


def _extract_verbatim_texts(entry: Dict[str, Any]) -> List[str]:
    texts: List[str] = []

    values = entry.get("verbatim_texts")
    if isinstance(values, list):
        texts.extend(str(item).strip() for item in values if str(item).strip())
    elif isinstance(values, str) and values.strip():
        texts.append(values.strip())

    single = entry.get("verbatim_text")
    if isinstance(single, str) and single.strip():
        texts.append(single.strip())

    seen: Set[str] = set()
    unique: List[str] = []
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _normalize_match_entry(entry: Dict[str, Any], lookup: Dict[str, Set[str]]) -> Dict[str, Any]:
    return {
        "refs": _resolve_refs(_extract_raw_refs(entry), lookup),
        "verbatim_texts": _extract_verbatim_texts(entry),
        "justification": str(entry.get("justification", "")).strip(),
    }


def _normalize_mutation_entry(entry: Dict[str, Any], lookup: Dict[str, Set[str]]) -> Dict[str, Any]:
    return {
        "refs": _resolve_refs(_extract_raw_refs(entry), lookup),
        "verbatim_texts": _extract_verbatim_texts(entry),
        "reason": str(entry.get("reason", "")).strip() or "Mutation detected.",
    }


def _normalize_omission_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "verbatim_texts": _extract_verbatim_texts(entry),
        "reason": str(entry.get("reason", "")).strip() or "Omission detected.",
    }


def _normalize_addition_entry(
    entry: Dict[str, Any],
    lookup: Dict[str, Set[str]],
    target_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    refs = _resolve_refs(_extract_raw_refs(entry), lookup)
    refs = [ref for ref in refs if _is_content_target(target_map.get(ref, {}))]
    return {
        "refs": refs,
        "reason": str(entry.get("reason", "")).strip() or "Addition detected.",
    }


def _split_multi_ref_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for entry in entries:
        refs = entry.get("refs", [])
        if len(refs) <= 1:
            out.append(entry)
            continue
        for ref in refs:
            out.append({
                "refs": [ref],
                "reason": entry.get("reason", "Addition detected."),
            })
    return out


def _build_uncovered_additions(
    matches: List[Dict[str, Any]],
    mutations: List[Dict[str, Any]],
    additions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    covered: Set[str] = set()
    for bucket in (matches, mutations, additions):
        for entry in bucket:
            covered.update(entry.get("refs", []))

    auto: List[Dict[str, Any]] = []
    for target in targets:
        ref_id = target.get("ref_id")
        if not ref_id or ref_id in covered:
            continue
        if not _is_content_target(target):
            continue
        auto.append({
            "refs": [ref_id],
            "reason": "No supporting verbatim portion was mapped from the input text.",
            "source_line": target.get("source_line"),
        })
    return auto


def _resolve_target_map(targets: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(target["ref_id"]): target
        for target in targets
        if target.get("ref_id")
    }


def _relation_count(entries: List[Dict[str, Any]], text_key: str) -> int:
    total = 0
    for entry in entries:
        refs = entry.get("refs", [])
        texts = entry.get(text_key, [])
        total += max(1, len(refs)) * max(1, len(texts))
    return total


def _count_elements(elements: Any) -> int:
    if isinstance(elements, pm.ParsedModel):
        return len(elements.nodes) + len(elements.guards) + len(elements.labels)
    if isinstance(elements, list):
        return len(elements)
    if isinstance(elements, dict):
        nested = elements.get("elements")
        return len(nested) if isinstance(nested, list) else 0
    return 0


def _calculate_metrics(
    matches: List[Dict[str, Any]],
    mutations: List[Dict[str, Any]],
    omissions: List[Dict[str, Any]],
    additions: List[Dict[str, Any]],
    input_text: Any,
    elements: Any,
    targets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    matched_refs = {ref for entry in matches for ref in entry.get("refs", [])}
    mutated_refs = {ref for entry in mutations for ref in entry.get("refs", [])}
    added_refs = {ref for entry in additions for ref in entry.get("refs", [])}
    covered_refs = matched_refs | mutated_refs | added_refs

    return {
        "match_count": len(matches),
        "mutation_count": len(mutations),
        "omission_count": len(omissions),
        "addition_count": len(additions),
        "match_relation_count": _relation_count(matches, "verbatim_texts"),
        "mutation_relation_count": _relation_count(mutations, "verbatim_texts"),
        "omission_relation_count": _relation_count(omissions, "verbatim_texts"),
        "unique_refs_in_matches": len(matched_refs),
        "unique_refs_in_mutations": len(mutated_refs),
        "unique_refs_in_additions": len(added_refs),
        "unique_refs_covered": len(covered_refs),
        "total_mappable_targets": len(targets),
        "input_text_length": len(input_text or ""),
        "element_count": _count_elements(elements),
    }


def _escape_hash_prefix(text: str) -> str:
    if not isinstance(text, str):
        return ""
    if text.startswith("#"):
        return "\\" + text
    return text


def _get_source_line(ref: str, target_map: Dict[str, Dict[str, Any]]) -> str:
    target = target_map.get(ref, {})
    source_line = str(target.get("source_line", "")).strip()
    return source_line or str(ref)


def format_mapping_details(result: MappingResult) -> str:
    target_map = _resolve_target_map(result.targets)
    out = ""

    out += "\n🔗 **Matched Elements:**\n\n"
    if result.matches:
        for match in result.matches:
            for ref in match.get("refs", []):
                out += f"  • Element: `{_escape_hash_prefix(_get_source_line(ref, target_map))}`\n"
            for text in match.get("verbatim_texts", []):
                out += f"  • Source Text: `{_escape_hash_prefix(text)}`\n"
            justification = match.get("justification", "")
            if justification:
                out += f"  • Justification: _{justification}_\n"
            out += "\n"
    else:
        out += "  _(No matches found)_\n\n"

    out += "\n🧪 **Mutations:**\n\n"
    if result.mutations:
        for mutation in result.mutations:
            for ref in mutation.get("refs", []):
                out += f"  • Element: `{_escape_hash_prefix(_get_source_line(ref, target_map))}`\n"
            for text in mutation.get("verbatim_texts", []):
                out += f"  • Source Text: `{_escape_hash_prefix(text)}`\n"
            out += f"  • Reason: _{mutation.get('reason', 'No reason provided')}_\n\n"
    else:
        out += "  _(No mutations found)_\n\n"

    out += "\n⚠️ **Omissions:**\n\n"
    if result.omissions:
        for omission in result.omissions:
            for text in omission.get("verbatim_texts", []):
                out += f"- `{_escape_hash_prefix(text)}`\n"
            out += f"  • Reason: _{omission.get('reason', 'No reason provided')}_\n\n"
    else:
        out += "  _(No omissions found)_\n\n"

    out += "\n➕ **Additions:**\n\n"
    if result.additions:
        for addition in result.additions:
            for ref in addition.get("refs", []):
                out += f"  • Element: `{_escape_hash_prefix(_get_source_line(ref, target_map))}`\n"
            out += f"  • Reason: _{addition.get('reason', 'No reason provided')}_\n\n"
    else:
        out += "  _(No additions found)_\n\n"

    return out
