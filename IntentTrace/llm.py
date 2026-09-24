from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import litellm
from pathlib import Path
from dotenv import load_dotenv
import json

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")


MODELS: Dict[str, Dict[str, Any]] = {


    "qwen": {
        "name": "Qwen 3.6",
        "model": "openai/qwen-3.6-27b-onprem",
        "api_base": os.getenv("QWEN_BASE"),
        "api_key": os.getenv("QWEN_API"),
        "chat_template_kwargs": {"enable_thinking": True},
        "temperature":1.0,
        "top_p":0.95,
        "top_k":20,
        "min_p":0.0,
        "presence_penalty":1.5,
        "repetition_penalty":1.0,
    },
    "gemini": {
        "name": "Gemini 3.1 Pro Preview",
        "model": "vertex_ai/gemini-3.1-pro-preview",
        "reasoning_effort": "medium",
        "api_base": os.getenv("GEMINI_BASE"),
        "api_key": os.getenv("GEMINI_API"),
    },

}


DEFAULT_MODEL = "gemini"


def resolve_model(identifier: str) -> str:
    if identifier in MODELS:
        return identifier
    for key, params in MODELS.items():
        if params["model"] == identifier or params["name"] == identifier:
            return key
    raise ValueError(f"Unknown model: {identifier}")


def _params_for(key: str, reasoning_effort: Optional[str] = None) -> Dict[str, Any]:
    params = {k: v for k, v in MODELS[key].items() if k != "name" and v is not None}
    if reasoning_effort:
        params["reasoning_effort"] = reasoning_effort
    return params


class LLMCallResult:

    def __init__(
        self,
        model_name: str,
        response: str,
        timestamp: Optional[str] = None,
        error: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        cost: Optional[float] = None,
        duration_seconds: Optional[float] = None,
        label: Optional[str] = None,
    ):
        self.model_name = model_name
        self.response = response
        self.timestamp = timestamp or datetime.now().isoformat()
        self.error = error
        self.reasoning_effort = reasoning_effort
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.reasoning_tokens = reasoning_tokens
        self.total_tokens = total_tokens
        self.cost = cost
        self.duration_seconds = duration_seconds
        self.label = label

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "model_name": self.model_name,
            "reasoning_effort": self.reasoning_effort,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "error": self.error,
        }


def _extract_usage_and_cost(response: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
        "cost": None,
    }

    usage = getattr(response, "usage", None)
    if usage is not None:
        info["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
        info["completion_tokens"] = getattr(usage, "completion_tokens", None)
        info["total_tokens"] = getattr(usage, "total_tokens", None)
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            info["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)

    try:
        hidden = getattr(response, "_hidden_params", None) or {}
        cost = hidden.get("response_cost")
        if cost is None:
            cost = litellm.completion_cost(completion_response=response)
        info["cost"] = cost
    except Exception:
        info["cost"] = None

    return info


class LLMCaller:

    def __init__(self, model: str, reasoning_effort: Optional[str] = None):
        self.key = resolve_model(model)
        self.params = _params_for(self.key, reasoning_effort)
        self.model_name = self.params["model"]
        self.reasoning_effort = self.params.get("reasoning_effort")
        self.history: List[LLMCallResult] = []

    async def call_async(
        self,
        messages: List[Dict[str, str]],
        label: Optional[str] = None,
        **kwargs,
    ) -> LLMCallResult:
        start = time.perf_counter()
        try:
            response = await asyncio.to_thread(
                litellm.completion, messages=messages, **self.params, **kwargs
            )
            duration = time.perf_counter() - start
            response_text = response.choices[0].message.content
            usage = _extract_usage_and_cost(response)
            result = LLMCallResult(
                model_name=self.model_name,
                response=response_text,
                reasoning_effort=self.reasoning_effort,
                duration_seconds=duration,
                label=label,
                **usage,
            )
        except Exception as exc:
            duration = time.perf_counter() - start
            result = LLMCallResult(
                model_name=self.model_name,
                response="",
                error=str(exc),
                reasoning_effort=self.reasoning_effort,
                duration_seconds=duration,
                label=label,
            )
        self.history.append(result)
        return result

    def get_cost_summary(self) -> Dict[str, Any]:

        def _sum(attr: str) -> Any:
            values = [getattr(r, attr) for r in self.history if getattr(r, attr) is not None]
            return sum(values) if values else None

        return {
            "model_name": self.model_name,
            "reasoning_effort": self.reasoning_effort,
            "total_calls": len(self.history),
            "total_errors": sum(1 for r in self.history if r.error),
            "total_prompt_tokens": _sum("prompt_tokens"),
            "total_completion_tokens": _sum("completion_tokens"),
            "total_reasoning_tokens": _sum("reasoning_tokens"),
            "total_tokens": _sum("total_tokens"),
            "total_cost": _sum("cost"),
            "calls": [r.to_dict() for r in self.history],
        }
