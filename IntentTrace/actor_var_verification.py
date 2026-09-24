from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from . import prompts
from . import parsed_model as pm


class VarActorValidationResult:

    def __init__(
        self,
        smc6: Dict[str, List[Dict[str, Any]]],
        smc7: Dict[str, List[Dict[str, Any]]],
        raw_response: Optional[str] = None,
        model_id: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        error: Optional[str] = None,
    ):
        self.smc6 = smc6
        self.smc7 = smc7
        self.raw_response = raw_response
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.error = error

    @property
    def smc6_passed(self) -> bool:
        return not self.smc6.get("mutations") and not self.smc6.get("omissions") and not self.smc6.get("additions")

    @property
    def smc7_passed(self) -> bool:
        return not self.smc7.get("mutations") and not self.smc7.get("omissions") and not self.smc7.get("additions")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "SMC6": self.smc6,
            "SMC6_passed": self.smc6_passed,
            "SMC7": self.smc7,
            "SMC7_passed": self.smc7_passed,
            "raw_response": self.raw_response,
            "model_id": self.model_id,
            "reasoning_effort": self.reasoning_effort,
            "error": self.error,
        }


def _empty_buckets() -> Dict[str, List[Dict[str, Any]]]:
    return {"matches": [], "mutations": [], "omissions": [], "additions": []}


def _normalize_buckets(raw: Any) -> Dict[str, List[Dict[str, Any]]]:
    out = _empty_buckets()
    if not isinstance(raw, dict):
        return out
    for key in out:
        value = raw.get(key)
        if isinstance(value, list):
            out[key] = [item for item in value if isinstance(item, dict)]
    return out


def _coerce_parsed_model(model: Any) -> pm.ParsedModel:
    if isinstance(model, pm.ParsedModel):
        return model
    if isinstance(model, dict):
        return pm.build_parsed_model(model)
    return pm.build_parsed_model({})


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        loaded = json.loads(cleaned)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start != -1 and end > start:
        try:
            loaded = json.loads(cleaned[start:end])
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            return None
    return None


async def validate_variables_and_actors(
    process_text: str,
    model: Any,
    llm_caller,
    model_id: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> VarActorValidationResult:

    if llm_caller is None:
        return VarActorValidationResult(
            smc6=_empty_buckets(),
            smc7=_empty_buckets(),
            error="No LLM caller available for SMC6/SMC7 validation.",
            model_id=model_id,
            reasoning_effort=reasoning_effort,
        )

    parsed_model = _coerce_parsed_model(model)
    prompt_template = prompts.VAR_ACTOR_MATCHING_PROMPT

    variable_records = pm.variable_payload(parsed_model)
    actor_records = pm.actor_payload(parsed_model)
    catalog_records = pm.node_catalog(parsed_model)

    def _render(records: List[Dict[str, Any]]) -> str:
        return json.dumps(records, indent=2) if records else "[]"

    prompt = prompt_template.replace("{{process_text}}", process_text or "")
    prompt = prompt.replace("{{variable_declarations}}", _render(variable_records))
    prompt = prompt.replace("{{actor_declarations}}", _render(actor_records))
    prompt = prompt.replace("{{node_catalog}}", _render(catalog_records))

    result = await llm_caller.call_async([{"role": "user", "content": prompt}], label="var_actor_validation")
    if result.error:
        return VarActorValidationResult(
            smc6=_empty_buckets(),
            smc7=_empty_buckets(),
            raw_response=str(result.error),
            error=str(result.error),
            model_id=model_id,
            reasoning_effort=reasoning_effort,
        )

    parsed = _extract_json_object(result.response or "") or {}
    variables_bucket = parsed.get("variables")
    actors_bucket = parsed.get("actors")

    return VarActorValidationResult(
        smc6=_normalize_buckets(variables_bucket),
        smc7=_normalize_buckets(actors_bucket),
        raw_response=result.response,
        model_id=model_id,
        reasoning_effort=reasoning_effort,
    )


def format_var_actor_validation(result: VarActorValidationResult) -> str:
    if result.error and not result.raw_response:
        return f"SMC6/SMC7 validation skipped: {result.error}\n"

    lines: List[str] = []
    for label, bucket in (("SMC6 (variables)", result.smc6), ("SMC7 (actors)", result.smc7)):
        passed = (
            not bucket.get("mutations")
            and not bucket.get("omissions")
            and not bucket.get("additions")
        )
        lines.append(f"{label}: {'PASS' if passed else 'FAIL'}")
        for category in ("matches", "mutations", "omissions", "additions"):
            entries = bucket.get(category, [])
            if not entries:
                continue
            lines.append(f"  {category} ({len(entries)}):")
            for entry in entries:
                refs = entry.get("refs") or []
                summary = str(entry.get("summary", "")).strip()
                rationale = entry.get("justification") or entry.get("reason") or ""
                if refs:
                    lines.append(f"    - {' | '.join(refs)}")
                elif summary:
                    lines.append(f"    - {summary}")
                if rationale:
                    lines.append(f"      reason: {rationale}")
    return "\n".join(lines) + "\n"
