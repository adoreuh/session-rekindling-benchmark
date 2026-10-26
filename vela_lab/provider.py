"""Model adapters.

Two rules govern this module.

**Failures are recorded, never laundered.**  A transport retry is fine; a retry
that hides that the first attempt may have reached the provider is not.  When
the outcome is unknown, the call is reported as indeterminate and the run is
marked accordingly — the alternative is a silent duplicate, which is worse than
a gap.

**The provider sees exactly the recorded payload.**  The loop serializes the
compiled request once, persists it, and hands *those bytes* to the transport.
No re-serialization step exists between "what we logged" and "what we sent".

Wall-clock time is used here for latency measurement only.  It never enters any
value the model can see, and the adapter never reads the virtual clock.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from .errors import ProviderError
from .jsonutil import canonical_json, json_copy

DEFAULT_TIMEOUT = 120.0
MAX_TRANSPORT_RETRIES = 2


def _mentions_lookup(text: str) -> bool:
    return any(token in text for token in ("查", "检索", "通知", "日期", "安排", "记得"))


@dataclass
class ModelResponse:
    """A normalized completion, plus everything needed to reconstruct the call.

    ``raw`` is the provider's own payload, kept verbatim.  A study that wants to
    explain an unexpected behaviour needs the original bytes — normalized fields
    discard exactly the provider-specific detail (finish-reason variants,
    refusal objects, safety metadata, cache hits) that turns out to matter after
    the fact.
    """

    content: str
    tool_calls: list
    finish_reason: str
    usage: dict
    model: str
    request_id: str = ""
    raw: dict | None = None
    latency_s: float | None = None
    attempts: int = 1
    system_fingerprint: str = ""
    attempt_log: list = field(default_factory=list)
    #: The provider's hidden reasoning text, when it returns one.
    #:
    #: Captured because some routes require it to be passed back on the next turn
    #: of a tool exchange: DeepSeek's thinking mode rejects a follow-up request
    #: whose assistant message omits it ("The `reasoning_content` in the thinking
    #: mode must be passed back to the API").  Without this the tool-continuation
    #: round fails for a reason that has nothing to do with the model's behaviour,
    #: which is exactly the kind of provider-shaped failure the harness must not
    #: silently record as a result.
    reasoning_content: str = ""

    def to_json(self) -> dict:
        return {
            "content": self.content,
            "reasoning_content": self.reasoning_content,
            "tool_calls": json_copy(self.tool_calls, "tool_calls"),
            "finish_reason": self.finish_reason,
            "usage": json_copy(self.usage, "usage"),
            "model": self.model,
            "request_id": self.request_id,
            "system_fingerprint": self.system_fingerprint,
            "latency_s": self.latency_s,
            "attempts": self.attempts,
            "attempt_log": json_copy(self.attempt_log, "attempt log"),
            "raw": json_copy(self.raw, "raw response") if self.raw is not None else None,
        }


class ModelAdapter:
    """Interface every adapter implements."""

    provider_id = "abstract"

    def complete(self, payload: dict) -> ModelResponse:  # pragma: no cover
        raise NotImplementedError

    def identity(self) -> dict:  # pragma: no cover
        raise NotImplementedError


class MockModel(ModelAdapter):
    """Deterministic offline model.

    Kept deliberately dumb: it emits a tool call when the user's text asks for
    one of the enabled tools, otherwise a fixed reply.  Its purpose is to
    exercise the pipeline and the audit — not to be plausible.  Results
    obtained with it are pipeline validation, never findings.
    """

    provider_id = "mock"

    def __init__(self, *, scripted: list | None = None, model_name: str = "mock"):
        self.scripted = list(scripted or [])
        self.model_name = model_name
        self.calls = 0

    def identity(self) -> dict:
        return {"provider": self.provider_id, "model": self.model_name}

    def complete(self, payload: dict) -> ModelResponse:
        self.calls += 1
        if self.scripted:
            item = self.scripted.pop(0)
            if isinstance(item, str):
                return ModelResponse(
                    content=item, tool_calls=[], finish_reason="stop",
                    usage={"prompt_tokens": 0, "completion_tokens": 0},
                    model=self.model_name,
                )
            return ModelResponse(
                content=item.get("content", ""),
                tool_calls=item.get("tool_calls", []),
                finish_reason=item.get("finish_reason", "stop"),
                usage=item.get("usage", {}),
                model=self.model_name,
            )

        messages = payload.get("messages", [])
        tools = {
            schema["function"]["name"] for schema in payload.get("tools", [])
        }
        last_user = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                last_user = message.get("content", "")
                break
        # A tool result is "just received" only if it is the latest message.
        # Earlier tool results belong to previous turns and do not mean this
        # round already consulted the notes.
        if messages and messages[-1].get("role") == "tool":
            return ModelResponse(
                content="mock answer: " + last_user[:40],
                tool_calls=[], finish_reason="stop",
                usage={"prompt_tokens": 0, "completion_tokens": 0},
                model=self.model_name,
            )
        # Baseline behaviour: consult notes on any substantive user turn.  This
        # makes the mock a "curious agent" so the exposure chain is exercised;
        # a mock that never retrieves cannot validate the pipeline it exists to
        # validate.
        substantive = len(last_user.strip()) >= 8
        if "search" in tools and (substantive or _mentions_lookup(last_user)):
            return ModelResponse(
                content="",
                tool_calls=[
                    {
                        "id": f"call_mock_{self.calls}",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": canonical_json(
                                {"query": last_user[:40], "top_k": 4}
                            ),
                        },
                    }
                ],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 0, "completion_tokens": 0},
                model=self.model_name,
            )
        return ModelResponse(
            content="mock answer: " + last_user[:40],
            tool_calls=[], finish_reason="stop",
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            model=self.model_name,
        )


class OpenAICompatibleModel(ModelAdapter):
    """POSTs to a ``/chat/completions`` endpoint.

    The API key is read from the environment by name and never stored in an
    artifact.  Only the *name* of the variable is recorded.
    """

    provider_id = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str = "VELA_API_KEY",
        timeout: float = DEFAULT_TIMEOUT,
        max_transport_retries: int = MAX_TRANSPORT_RETRIES,
        transport: Callable | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ):
        if not base_url or not base_url.startswith(("http://", "https://")):
            raise ProviderError(f"base_url must be an http(s) URL, got {base_url!r}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_transport_retries = max_transport_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._transport = transport or self._http_post
        self.attempt_log: list = []

    def identity(self) -> dict:
        return {
            "provider": self.provider_id,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def complete(self, payload: dict) -> ModelResponse:
        body = dict(payload)
        body.setdefault("model", self.model)
        if self.temperature is not None:
            body.setdefault("temperature", self.temperature)
        if self.max_tokens is not None:
            body.setdefault("max_tokens", self.max_tokens)
        encoded = canonical_json(body).encode("utf-8")

        attempts = 0
        last_error: Exception | None = None
        total_started = time.monotonic()
        while attempts <= self.max_transport_retries:
            attempts += 1
            started = time.monotonic()
            try:
                raw = self._transport(encoded)
                elapsed = time.monotonic() - started
                self.attempt_log.append({"attempt": attempts, "ok": True,
                                         "latency_s": round(elapsed, 3)})
                if attempts > 1:
                    raise ProviderError(
                        "a retry succeeded after the first attempt may already "
                        "have reached the provider; the outcome of the earlier "
                        "attempt is indeterminate",
                        attempts=attempts,
                        retryable=False,
                    )
                return self._parse(
                    raw,
                    latency_s=round(time.monotonic() - total_started, 3),
                    attempts=attempts,
                )
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                elapsed = time.monotonic() - started
                last_error = exc
                self.attempt_log.append(
                    {"attempt": attempts, "ok": False,
                     "latency_s": round(elapsed, 3), "error": type(exc).__name__}
                )
            except ProviderError:
                raise

        raise ProviderError(
            f"transport failed after {attempts} attempt(s): {type(last_error).__name__}",
            attempts=attempts,
            retryable=True,
        )

    def _http_post(self, encoded: bytes) -> dict:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ProviderError(
                f"environment variable {self.api_key_env} is not set; refusing to "
                "call the provider without credentials"
            )
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=encoded,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise ProviderError(
                f"HTTP {exc.code} from provider: {detail}", retryable=False
            ) from exc

    def _parse(self, raw: dict, *, latency_s: float | None = None,
               attempts: int = 1) -> ModelResponse:
        if not isinstance(raw, dict):
            raise ProviderError("provider returned a non-object payload")
        try:
            choice = raw["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"provider payload has no choices: {raw}") from exc
        message = choice.get("message", {})
        tool_calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            arguments = function.get("arguments", "{}")
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                except json.JSONDecodeError:
                    parsed = {"__malformed__": arguments}
            else:
                parsed = arguments
            tool_calls.append(
                {
                    "id": call.get("id", ""),
                    "type": "function",
                    "function": {"name": function.get("name", ""),
                                 "arguments": parsed},
                }
            )
        # Providers disagree on the field name, so all known spellings are read
        # rather than one being assumed.  A missed spelling shows up as a failed
        # continuation, which is an expensive way to learn a field name.
        reasoning = ""
        for key in ("reasoning_content", "reasoning", "thinking"):
            value = message.get(key)
            if isinstance(value, str) and value:
                reasoning = value
                break
        return ModelResponse(
            content=message.get("content") or "",
            reasoning_content=reasoning,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", ""),
            usage=raw.get("usage", {}),
            model=raw.get("model", self.model),
            request_id=raw.get("id", ""),
            raw=raw,
            latency_s=latency_s,
            attempts=attempts,
            system_fingerprint=raw.get("system_fingerprint", "") or "",
            attempt_log=list(self.attempt_log),
        )
