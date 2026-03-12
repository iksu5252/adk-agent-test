"""Google ADK-style LLM gateway.

This module provides an orchestration layer inspired by Google ADK flow:
- route request to internal model or ChatGPT-compatible model
- keep transport/model client pluggable
- parse/validate structured JSON outputs with normalization rules
- expose hooks for future feature expansion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence


class Backend(str, Enum):
    """Model backend selector."""

    INTERNAL = "internal"
    CHATGPT = "chatgpt"


class GatewayError(RuntimeError):
    """Raised for gateway-level failures."""


class ValidationError(ValueError):
    """Raised when structured output parsing/validation fails."""


class ModelClient(Protocol):
    """Google ADK-like model client abstraction.

    In real ADK projects, this can wrap model invocation from an ADK Agent/Runner.
    """

    def generate(self, *, messages: Sequence[Dict[str, str]], system_instruction: Optional[str] = None) -> str:
        """Return the model output text."""


Rule = Callable[[str, Any], Any]


@dataclass(slots=True)
class JsonOutputParser:
    """Parse/normalize/validate JSON object payload.

    - aliases: incorrect or legacy key -> canonical key
    - required_keys: keys required after alias normalization
    - rules: key-based coercion/validation function map
    - allow_unknown: whether keys without rules are allowed
    """

    aliases: Mapping[str, str] = field(default_factory=dict)
    required_keys: Iterable[str] = field(default_factory=tuple)
    rules: Mapping[str, Rule] = field(default_factory=dict)
    allow_unknown: bool = False

    def parse(self, payload: str) -> Dict[str, Any]:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Invalid JSON payload: {exc}") from exc

        if not isinstance(data, dict):
            raise ValidationError("Model output JSON must be an object.")

        normalized: Dict[str, Any] = {}
        for key, value in data.items():
            canonical = self.aliases.get(key, key)
            normalized[canonical] = value

        missing = [key for key in self.required_keys if key not in normalized]
        if missing:
            raise ValidationError(f"Missing required keys: {', '.join(missing)}")

        validated: Dict[str, Any] = {}
        for key, value in normalized.items():
            rule = self.rules.get(key)
            if rule is None:
                if self.allow_unknown:
                    validated[key] = value
                    continue
                raise ValidationError(f"Unknown key not allowed: {key}")

            try:
                validated[key] = rule(key, value)
            except ValidationError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise ValidationError(f"Validation failed for '{key}': {exc}") from exc

        return validated


# ---------- reusable validation rules ----------


def require_type(expected: type) -> Rule:
    def _rule(key: str, value: Any) -> Any:
        if not isinstance(value, expected):
            raise ValidationError(f"Key '{key}' expected {expected.__name__}, got {type(value).__name__}")
        return value

    return _rule



def one_of(allowed: Iterable[Any]) -> Rule:
    values = set(allowed)

    def _rule(key: str, value: Any) -> Any:
        if value not in values:
            raise ValidationError(f"Key '{key}' has invalid value '{value}'. Allowed: {sorted(values)}")
        return value

    return _rule



def to_int(min_value: Optional[int] = None, max_value: Optional[int] = None) -> Rule:
    def _rule(key: str, value: Any) -> int:
        try:
            ivalue = int(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Key '{key}' cannot be converted to int") from exc

        if min_value is not None and ivalue < min_value:
            raise ValidationError(f"Key '{key}' must be >= {min_value}")
        if max_value is not None and ivalue > max_value:
            raise ValidationError(f"Key '{key}' must be <= {max_value}")
        return ivalue

    return _rule


PreHook = Callable[[Sequence[Dict[str, str]]], Sequence[Dict[str, str]]]
PostHook = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass(slots=True)
class AdkLlmGateway:
    """Gateway orchestrator for Google ADK-based service layers."""

    internal_client: ModelClient
    chatgpt_client: ModelClient
    parser: JsonOutputParser
    backend: Backend = Backend.INTERNAL
    chatgpt_enabled: bool = True
    system_instruction: Optional[str] = None
    pre_hooks: List[PreHook] = field(default_factory=list)
    post_hooks: List[PostHook] = field(default_factory=list)

    def set_backend(self, backend: Backend) -> None:
        self.backend = backend

    def set_chatgpt_enabled(self, enabled: bool) -> None:
        self.chatgpt_enabled = enabled

    def request(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        backend: Optional[Backend] = None,
    ) -> Dict[str, Any]:
        target = backend or self.backend

        if target == Backend.CHATGPT and not self.chatgpt_enabled:
            raise GatewayError("ChatGPT backend is disabled (off).")

        outbound = messages
        for hook in self.pre_hooks:
            outbound = hook(outbound)

        client = self.internal_client if target == Backend.INTERNAL else self.chatgpt_client
        try:
            raw = client.generate(messages=outbound, system_instruction=self.system_instruction)
        except Exception as exc:  # noqa: BLE001
            raise GatewayError(f"Backend request failed ({target.value}): {exc}") from exc

        parsed = self.parser.parse(raw)

        for hook in self.post_hooks:
            parsed = hook(parsed)

        return parsed


def build_default_parser() -> JsonOutputParser:
    """Default strict parser for common intent classification style outputs."""

    return JsonOutputParser(
        aliases={"conf": "confidence", "result": "answer"},
        required_keys=("intent", "confidence", "answer"),
        rules={
            "intent": one_of({"search", "chat", "command"}),
            "confidence": to_int(0, 100),
            "answer": require_type(str),
        },
        allow_unknown=False,
    )
