"""Provider qualification gate: decide whether a route may be used for real runs.

Rationale. A gateway that answers plain questions is not automatically usable for
this research.  The harness needs a route to (a) accept a system prompt and tool
schemas, (b) emit parseable tool calls, (c) return a *complete* answer rather than
one truncated at a reasoning budget, and (d) behave consistently enough across
repeats that a difference between branches is about the branch and not the
transport.  Each of those can fail independently, and a failure shows up as a
plausible-looking model behaviour if it is not screened out first.

The most instructive failure found so far: ``hy3`` is a reasoning model, so with a
small ``max_tokens`` it spends the whole budget on reasoning tokens and returns
empty content with ``finish_reason="length"``.  Read naively that looks like a
model that "said nothing", and it would enter the results as a behaviour.  The
gate exists so that class of artifact is rejected at the door.

Usage::

    VELA_API_KEY=... python -m tools.provider_gate --models hy3 Qwen3.5-397B-A17B
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_BASE_URL = "__REDACTED_BASE_URL__"

# Reasoning routes need headroom or they answer nothing.  The value is a
# screening parameter, not an experimental one; the experiment's own budget is
# set separately and recorded in the frozen policy.
SCREEN_MAX_TOKENS = 1536

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "world_time",
            "description": "Get the current date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search your notes and documents for relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for."},
                    "top_k": {"type": "integer", "description": "How many results."},
                },
                "required": ["query"],
            },
        },
    },
]


@dataclass
class Attempt:
    ok: bool
    latency_s: float
    finish_reason: str = ""
    content_chars: int = 0
    reasoning_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    tool_calls: list = field(default_factory=list)
    error: str = ""


@dataclass
class RouteReport:
    model: str
    checks: dict = field(default_factory=dict)
    attempts: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def qualified(self) -> bool:
        return all(self.checks.values()) and bool(self.checks)

    def to_json(self) -> dict:
        latencies = [a.latency_s for a in self.attempts if a.ok]
        return {
            "model": self.model,
            "qualified": self.qualified,
            "checks": dict(self.checks),
            "attempts": [
                {
                    "ok": a.ok,
                    "latency_s": a.latency_s,
                    "finish_reason": a.finish_reason,
                    "content_chars": a.content_chars,
                    "reasoning_tokens": a.reasoning_tokens,
                    "completion_tokens": a.completion_tokens,
                    "prompt_tokens": a.prompt_tokens,
                    "tool_calls": a.tool_calls,
                    "error": a.error,
                }
                for a in self.attempts
            ],
            "latency_median_s": round(statistics.median(latencies), 2) if latencies else None,
            "notes": self.notes,
        }


class Gateway:
    def __init__(self, *, base_url: str, api_key: str, timeout: float = 240.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, body: dict) -> tuple[dict, float]:
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        return payload, round(time.monotonic() - started, 3)


def _parse_attempt(payload: dict, latency: float) -> Attempt:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message", {}) or {}
    usage = payload.get("usage", {}) or {}
    tool_calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function", {}) or {}
        args = function.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"__malformed__": args}
        tool_calls.append({"name": function.get("name", ""), "arguments": args})
    return Attempt(
        ok=True,
        latency_s=latency,
        finish_reason=choice.get("finish_reason", ""),
        content_chars=len(message.get("content") or ""),
        reasoning_tokens=(usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens", 0
        ),
        completion_tokens=usage.get("completion_tokens", 0),
        prompt_tokens=usage.get("prompt_tokens", 0),
        tool_calls=tool_calls,
    )


def qualify(gateway: Gateway, model: str, *, repeats: int = 3) -> RouteReport:
    report = RouteReport(model=model)

    # 1. A plain completion finishes and produces text.
    try:
        payload, latency = gateway.complete(
            {
                "model": model,
                "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
                "max_tokens": SCREEN_MAX_TOKENS,
            }
        )
        attempt = _parse_attempt(payload, latency)
        report.attempts.append(attempt)
        report.checks["text_completion"] = attempt.content_chars > 0
        report.checks["not_truncated"] = attempt.finish_reason == "stop"
        if attempt.reasoning_tokens and attempt.finish_reason == "length":
            report.notes.append(
                f"reasoning route: {attempt.reasoning_tokens} reasoning tokens consumed "
                f"the budget at max_tokens={SCREEN_MAX_TOKENS}"
            )
        report.checks["usage_reported"] = attempt.prompt_tokens > 0
    except Exception as exc:
        report.checks["text_completion"] = False
        report.checks["not_truncated"] = False
        report.checks["usage_reported"] = False
        report.notes.append(f"plain completion failed: {exc}")

    # 2. Tool schemas are accepted and a tool call comes back parseable.
    try:
        payload, latency = gateway.complete(
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": "What is today's date? Use a tool."}
                ],
                "tools": TOOLS,
                "tool_choice": "auto",
                "max_tokens": SCREEN_MAX_TOKENS,
            }
        )
        attempt = _parse_attempt(payload, latency)
        report.attempts.append(attempt)
        names = [call["name"] for call in attempt.tool_calls]
        report.checks["emits_tool_call"] = bool(attempt.tool_calls)
        report.checks["tool_name_known"] = all(
            name in {"world_time", "search"} for name in names
        )
        report.checks["tool_args_parseable"] = all(
            "__malformed__" not in call["arguments"] for call in attempt.tool_calls
        )
        if not attempt.tool_calls:
            report.notes.append("no tool call returned; the route may not support tools")
    except Exception as exc:
        report.checks["emits_tool_call"] = False
        report.checks["tool_name_known"] = False
        report.checks["tool_args_parseable"] = False
        report.notes.append(f"tool call round trip failed: {exc}")

    # 3. A tool result can be fed back and the route answers.
    try:
        payload, latency = gateway.complete(
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": "What is today's date?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_gate_1",
                                "type": "function",
                                "function": {"name": "world_time", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_gate_1",
                        "name": "world_time",
                        "content": json.dumps({"current_time": "2026-03-11T09:00:00+08:00"}),
                    },
                ],
                "tools": TOOLS,
                "max_tokens": SCREEN_MAX_TOKENS,
            }
        )
        attempt = _parse_attempt(payload, latency)
        report.attempts.append(attempt)
        report.checks["consumes_tool_result"] = (
            attempt.content_chars > 0 and attempt.finish_reason == "stop"
        )
    except Exception as exc:
        report.checks["consumes_tool_result"] = False
        report.notes.append(f"tool-result continuation failed: {exc}")

    # 4. Repeat stability: same prompt, same observable outcome class.
    outcomes = []
    for _ in range(repeats):
        try:
            payload, latency = gateway.complete(
                {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Answer with one word: is a visa a document?",
                        }
                    ],
                    "max_tokens": SCREEN_MAX_TOKENS,
                }
            )
            attempt = _parse_attempt(payload, latency)
            report.attempts.append(attempt)
            outcomes.append(
                (attempt.finish_reason == "stop") and attempt.content_chars > 0
            )
        except Exception as exc:
            outcomes.append(False)
            report.notes.append(f"stability attempt failed: {exc}")
    report.checks["repeat_stable"] = bool(outcomes) and all(outcomes)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="VELA_API_KEY")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    key = os.environ.get(args.api_key_env)
    if not key:
        print(f"environment variable {args.api_key_env} is not set")
        return 2

    gateway = Gateway(base_url=args.base_url, api_key=key)
    reports = []
    for model in args.models:
        print(f"\n=== {model} ===", flush=True)
        report = qualify(gateway, model, repeats=args.repeats)
        reports.append(report.to_json())
        for check, passed in sorted(report.checks.items()):
            print(f"  {'PASS' if passed else 'FAIL'}  {check}")
        for note in report.notes:
            print(f"  note: {note}")
        attempts = report.to_json()["attempts"]
        ok = [a for a in attempts if a["ok"]]
        if ok:
            print(f"  latency median: {report.to_json()['latency_median_s']}s "
                  f"over {len(ok)} calls")
        print(f"  QUALIFIED: {report.qualified}")

    summary = {
        "base_url": args.base_url,
        "models": reports,
        "qualified_models": [r["model"] for r in reports if r["qualified"]],
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(
        {r["model"]: r["qualified"] for r in reports}, indent=2, ensure_ascii=False
    ))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"wrote {args.out}")
    return 0 if summary["qualified_models"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
