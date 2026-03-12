"""Google ADK-style LLM gateway (rebuilt).

Goals:
- Route requests to internal or ChatGPT backend.
- Allow ChatGPT on/off at runtime.
- Inject agent description into system context every request.
- Parse and validate JSON output robustly.
- Auto-fix common field corruption cases (merged/swapped ID/group).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import re
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence


class Backend(str, Enum):
    INTERNAL = "internal"
    CHATGPT = "chatgpt"


class GatewayError(RuntimeError):
    """Backend execution error."""


class ValidationError(ValueError):
    """Structured output validation error."""


class ModelClient(Protocol):
    """Minimal ADK-friendly model client protocol."""

    def generate(self, *, messages: Sequence[Dict[str, str]], system_instruction: Optional[str] = None) -> str:
        ...


Rule = Callable[[str, Any], Any]
Normalizer = Callable[[Dict[str, Any]], Dict[str, Any]]
PreHook = Callable[[Sequence[Dict[str, str]]], Sequence[Dict[str, str]]]
PostHook = Callable[[Dict[str, Any]], Dict[str, Any]]


def compose_system_instruction(base_instruction: Optional[str], agent_description: Optional[str]) -> Optional[str]:
    """Merge base instruction + agent description into one system instruction."""
    parts: list[str] = []
    if base_instruction:
        parts.append(base_instruction.strip())
    if agent_description:
        parts.append(f"Agent description:\n{agent_description.strip()}")
    return "\n\n".join(parts) if parts else None


@dataclass(slots=True)
class JsonOutputParser:
    """JSON object parser with aliasing, normalizing and per-key rules."""

    aliases: Mapping[str, str] = field(default_factory=dict)
    required_keys: Sequence[str] = field(default_factory=tuple)
    normalizers: Sequence[Normalizer] = field(default_factory=tuple)
    rules: Mapping[str, Rule] = field(default_factory=dict)
    allow_unknown: bool = False

    def parse(self, payload: str) -> Dict[str, Any]:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Invalid JSON payload: {exc}") from exc

        if not isinstance(data, dict):
            raise ValidationError("Model output must be a JSON object.")

        normalized: Dict[str, Any] = {}
        for k, v in data.items():
            normalized[self.aliases.get(k, k)] = v

        for normalize in self.normalizers:
            normalized = normalize(normalized)

        missing = [k for k in self.required_keys if k not in normalized]
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


def require_type(expected_type: type) -> Rule:
    def _rule(key: str, value: Any) -> Any:
        if not isinstance(value, expected_type):
            raise ValidationError(f"Key '{key}' expected {expected_type.__name__}, got {type(value).__name__}")
        return value

    return _rule


def one_of(allowed_values: Sequence[Any]) -> Rule:
    allowed = set(allowed_values)

    def _rule(key: str, value: Any) -> Any:
        if value not in allowed:
            raise ValidationError(f"Key '{key}' has invalid value '{value}'. Allowed: {sorted(allowed)}")
        return value

    return _rule


def to_int(min_value: Optional[int] = None, max_value: Optional[int] = None) -> Rule:
    def _rule(key: str, value: Any) -> int:
        try:
            n = int(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Key '{key}' cannot be converted to int") from exc

        if min_value is not None and n < min_value:
            raise ValidationError(f"Key '{key}' must be >= {min_value}")
        if max_value is not None and n > max_value:
            raise ValidationError(f"Key '{key}' must be <= {max_value}")
        return n

    return _rule


def match_pattern(pattern: str) -> Rule:
    regex = re.compile(pattern)

    def _rule(key: str, value: Any) -> str:
        if not isinstance(value, str):
            raise ValidationError(f"Key '{key}' expected str, got {type(value).__name__}")
        if not regex.fullmatch(value):
            raise ValidationError(f"Key '{key}' has invalid format: {value}")
        return value

    return _rule


def fix_identity_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Repair frequent ID/group corruption.

    1) Merged form: ID='ABCD22DD.BB_G', group='1' -> split.
    2) Swapped form: ID='BB_G', group='ABCD22DD' -> swap.
    """
    fixed = dict(data)

    id_value = fixed.get("ID")
    group_value = fixed.get("group")

    if isinstance(id_value, str) and "." in id_value:
        left, right = id_value.split(".", 1)
        if left and right:
            fixed["ID"] = left
            fixed["group"] = right

    id_value = fixed.get("ID")
    group_value = fixed.get("group")

    looks_like_group = isinstance(id_value, str) and bool(re.fullmatch(r"[A-Za-z]+_[A-Za-z0-9]+", id_value))
    looks_like_id = isinstance(group_value, str) and bool(re.fullmatch(r"[A-Za-z0-9]{6,}", group_value))

    if looks_like_group and looks_like_id:
        fixed["ID"], fixed["group"] = fixed["group"], fixed["ID"]

    return fixed


@dataclass(slots=True)
class AdkLlmGateway:
    """Gateway for backend routing + system-context composition + JSON parsing."""

    internal_client: ModelClient
    chatgpt_client: ModelClient
    parser: JsonOutputParser
    backend: Backend = Backend.INTERNAL
    chatgpt_enabled: bool = True
    system_instruction: Optional[str] = None
    agent_description: Optional[str] = None
    pre_hooks: Sequence[PreHook] = field(default_factory=tuple)
    post_hooks: Sequence[PostHook] = field(default_factory=tuple)

    def set_backend(self, backend: Backend) -> None:
        self.backend = backend

    def set_chatgpt_enabled(self, enabled: bool) -> None:
        self.chatgpt_enabled = enabled

    def set_agent_description(self, description: Optional[str]) -> None:
        self.agent_description = description

    def request(self, messages: Sequence[Dict[str, str]], *, backend: Optional[Backend] = None) -> Dict[str, Any]:
        selected = backend or self.backend
        if selected == Backend.CHATGPT and not self.chatgpt_enabled:
            raise GatewayError("ChatGPT backend is disabled.")

        outbound = messages
        for hook in self.pre_hooks:
            outbound = hook(outbound)

        client = self.internal_client if selected == Backend.INTERNAL else self.chatgpt_client
        instruction = compose_system_instruction(self.system_instruction, self.agent_description)

        try:
            raw = client.generate(messages=outbound, system_instruction=instruction)
        except Exception as exc:  # noqa: BLE001
            raise GatewayError(f"Backend request failed ({selected.value}): {exc}") from exc

        parsed = self.parser.parse(raw)

        for hook in self.post_hooks:
            parsed = hook(parsed)

        return parsed


def build_default_parser() -> JsonOutputParser:
    return JsonOutputParser(
        aliases={"conf": "confidence", "result": "answer"},
        required_keys=("intent", "confidence", "answer"),
        rules={
            "intent": one_of(("search", "chat", "command")),
            "confidence": to_int(0, 100),
            "answer": require_type(str),
        },
        allow_unknown=False,
    )


def build_identity_parser() -> JsonOutputParser:
    return JsonOutputParser(
        required_keys=("ID", "group", "age"),
        normalizers=(fix_identity_fields,),
        rules={
            "ID": match_pattern(r"[A-Za-z0-9]{6,}"),
            "group": match_pattern(r"[A-Za-z]+_[A-Za-z0-9]+"),
            "age": to_int(0, 130),
        },
        allow_unknown=False,
    )
