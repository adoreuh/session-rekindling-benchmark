"""The minimal agent turn loop.

Deliberately small.  What it does:

1. check budgets, state and outstanding tool results;
2. compile the request from observations (via the compiler — the only outlet);
3. persist the request, then dispatch the recorded payload;
4. record the response and dispatch on its finish reason;
5. validate tool calls, execute them through the tool belt, record both the raw
   and visible sides;
6. stop at a termination condition, without quietly patching anything up.

What it deliberately does *not* do: auto-summarize context, auto-repair
malformed output, add tools mid-turn, retry silently, or maintain its own
memory.  Each of those changes what the model saw, and therefore has to be an
event rather than a convenience.

Failure policy.  A business-level tool error is returned to the model as an
ordinary error result — that is what a real product would do.  A harness-level
failure (a contract violation, an evidence-injection failure, an
inconsistent state) terminates the run and is recorded as a failure, never
handed to the model to work around.  Conflating the two is how "the model
failed" gets confused with "the harness broke".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .compiler import CompiledRequest, RequestCompiler
from .errors import (
    ContractViolation,
    GateRejection,
    ProviderError,
    ValidationError,
)
from .jsonutil import canonical_json, digest as _digest, json_copy
from .provider import ModelAdapter, ModelResponse
from .tools import ToolBelt

MAX_TOOL_ROUNDS_DEFAULT = 4


@dataclass
class TurnOutcome:
    """What one agent turn produced."""

    request_count: int
    tool_calls: int
    assistant_text: str
    finish_reasons: list
    failure: dict | None = None
    indeterminate: bool = False
    search: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "request_count": self.request_count,
            "tool_calls": self.tool_calls,
            "assistant_text": self.assistant_text,
            "finish_reasons": list(self.finish_reasons),
            "failure": self.failure,
            "indeterminate": self.indeterminate,
            "search": dict(self.search),
        }


class AgentLoop:
    """Runs agent turns against a session's log.

    ``session`` is duck-typed to avoid a circular import: it must expose
    ``append(type, data)``, ``state``, ``clock``, ``observations``,
    ``request_index``.
    """

    def __init__(
        self,
        *,
        session: Any,
        model: ModelAdapter,
        compiler: RequestCompiler,
        tools: ToolBelt,
        budget: int = MAX_TOOL_ROUNDS_DEFAULT,
        request_params: dict | None = None,
        request_gate: Any = None,
    ):
        self.session = session
        self.model = model
        self.compiler = compiler
        self.tools = tools
        self.budget = budget
        self.request_params = dict(request_params or {})
        # Optional check on the exact request about to leave the harness.  Typed
        # as a bare callable on purpose: the check is a property of the
        # *scenario's* exposure contract, and the kernel must not import it.
        self.request_gate = request_gate
        self.search_requested = 0
        self.search_executed = 0
        self.search_evidence_delivered = 0

    def run_turn(self, *, label: str = "") -> TurnOutcome:
        self.session.append("turn.started", {"label": label})
        self.session.append("turn.context", self._turn_context(label))

        requests = 0
        tool_calls = 0
        reasons: list = []
        assistant_text = ""

        for round_index in range(self.budget + 1):
            compiled = self._compile(requests)
            requests += 1
            self._check_request_gate(compiled)
            response = self._dispatch(compiled)
            reasons.append(response.finish_reason)

            self.session.append(
                "model.response",
                {
                    "request_index": compiled.request_index,
                    "response": response.to_json(),
                },
            )

            if response.finish_reason == "length":
                return self._fail(
                    "response truncated by the token limit; a truncated answer is "
                    "not a behaviour observation",
                    requests, tool_calls, reasons, assistant_text,
                )

            if response.tool_calls:
                if round_index >= self.budget:
                    return self._fail(
                        f"tool rounds exceeded the budget of {self.budget}",
                        requests, tool_calls, reasons, assistant_text,
                    )
                # Record the model's own tool-requesting turn before executing
                # anything.  The projection has to replay it, or the tool results
                # that follow answer a message that is not in the conversation —
                # a sequence strict providers reject outright and tolerant ones
                # accept while the model sees a result arrive from nowhere.
                self._record_assistant_tool_call(response)
                tool_calls += self._handle_tool_calls(response.tool_calls)
                continue

            assistant_text = response.content
            self._record_assistant(response.content)
            self.session.append(
                "turn.completed",
                {"requests": requests, "tool_calls": tool_calls,
                 "finish_reasons": reasons},
            )
            # Close the user turn: turn-scoped notes stop being active.  They
            # remain recorded as previously exposed, so a reader can still see
            # that exposure happened without the evidence continuing to reach
            # the model.
            self.session.observations.end_turn()
            self.session.append(
                "observation.retired",
                {"retired": len(self.session.observations.expired_observations()),
                 "active": len(self.session.observations.fragments())},
            )
            outcome = TurnOutcome(requests, tool_calls, assistant_text, reasons)
            outcome.search = self.search_accounting()
            return outcome

        return self._fail(
            f"turn did not terminate within the tool budget of {self.budget}",
            requests, tool_calls, reasons, assistant_text,
        )

    # ------------------------------------------------------ do-interventions

    def inject_tool_call(self, tool_name: str, arguments: dict | None = None,
                         *, call_id: str | None = None,
                         label: str = "forced") -> object:
        """Inject a completed tool call into model-visible history.

        A do-intervention on the model-visible state: the transcript gains the
        assistant tool-call and the tool result exactly where a real exchange
        would put them, while the user turn is NOT touched.  ``do(U=1)`` is
        therefore distinguishable from "the user asked for a check" — every
        record carries ``injected=<label>`` so a later reader can separate the
        experimenter's intervention from model behaviour.
        """
        from .jsonutil import canonical_json
        arguments = dict(arguments or {})
        call_id = call_id or f"{label}-{tool_name}-1"

        # The assistant tool-requesting turn, normalized to the wire shape a
        # provider would accept: the tool result that follows must answer a
        # message that is actually in the conversation.
        normalized = [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": canonical_json(arguments),
            },
        }]
        seq = self.session.head_seq() + 1
        fragment = self.session.observations.assistant_tool_call(
            tool_calls=normalized, event_seq=seq, text="", reasoning="",
        )
        self.session.append(
            "observation.compiled",
            {
                "kind": fragment.kind,
                "text": fragment.text,
                "tool_calls": normalized,
                "reasoning_content": "",
                "injected": label,
                "provenance": fragment.provenance.to_json(),
            },
        )

        self.session.append(
            "tool.call",
            {"call_id": call_id, "tool_name": tool_name,
             "arguments": arguments, "injected": label},
        )
        result = self.tools.execute(
            call_id, tool_name, arguments, clock=self.session.clock
        )
        self.session.append(
            "tool.raw_result",
            {"call_id": call_id, "result": result.raw, "injected": label},
        )
        self.session.append(
            "tool.visible_result",
            {"call_id": call_id, "result": result.visible, "injected": label},
        )

        text = canonical_json(result.visible)
        seq = self.session.head_seq() + 1
        fragment = self.session.observations.tool_result(
            text, call_id=call_id, event_seq=seq,
            meta={"call_id": call_id, "tool_name": tool_name,
                  "injected": label},
        )
        self.session.append(
            "observation.compiled",
            {
                "kind": fragment.kind,
                "call_id": call_id,
                "tool_name": tool_name,
                "text": text,
                "injected": label,
                "provenance": fragment.provenance.to_json(),
            },
        )
        return result

    # ------------------------------------------------------------- internals

    def search_accounting(self) -> dict:
        """The four levels of search activity, kept apart.

        Counting only the last level conflates willingness with capability: a model
        that asked to search and was refused looks identical to one that never
        asked.  Only ``search_requested`` is about the model; the other two are
        about whether the harness let the attempt through.
        """
        return {
            "search_requested": self.search_requested,
            "search_executed": self.search_executed,
            "search_evidence_delivered": self.search_evidence_delivered,
        }

    def _turn_context(self, label: str) -> dict:
        """Snapshot the policy in force for this turn.

        Borrowed from Codex's per-turn ``TurnContextItem``.  Its value for a
        longitudinal study is that a later reader can tell *which* policy
        produced a turn without inferring it from the surrounding events.  If a
        branch's exposure policy, tool set or prompt ever changes between turns,
        the change is visible at the turn boundary rather than buried.
        """
        tools = self.tools
        return {
            "label": label,
            "turn_index": self.session.state.turn_index + 1,
            "virtual_time": self.session.clock.instant,
            "time_zone": self.session.clock.time_zone,
            "prompt_id": self.compiler.bundle.prompt_id,
            "prompt_digest": self.compiler.bundle.digest(),
            "enabled_tools": list(tools.enabled),
            "tool_schema_digest": _digest(tools.schemas()),
            "exposure": tools.exposure.to_json(),
            "exposure_digest": tools.exposure.digest(),
            "budget": self.budget,
            "request_params": json_copy(self.request_params, "request params"),
            "model": self.model.identity(),
        }

    def _compile(self, requests_so_far: int) -> CompiledRequest:
        return self.compiler.compile(
            self.session.observations,
            request_index=self.session.next_request_index(),
            bound_at_seq=self.session.head_seq(),
            extra_params=self.request_params,
        )

    def _check_request_gate(self, compiled: CompiledRequest) -> None:
        """Check the serialized request before it leaves the harness.

        Positioned here rather than at configuration time because everything the
        gate is supposed to catch happens *after* configuration: the memory
        wrapper, the folding of notes into the user turn, the tool-schema list
        sent alongside, and any inherited history all materialize during
        compilation.  A check that ran on the policy would inspect the intention
        rather than the payload.

        Fail-closed, with the scope preserved.  A request-level rejection
        invalidates this branch; a batch-level one invalidates every branch
        sharing the parent state, and the runner is told which it is so it does
        not report a shared defect as one branch's outcome.
        """
        if self.request_gate is None:
            return
        verdict = self.request_gate(compiled)
        # Recorded for every request, not only failing ones: a later reader has
        # to be able to see that the check ran and completed, since a run with no
        # gate events is indistinguishable from one whose gate found nothing.
        self.session.append("exposure.checked", verdict.to_json())
        status = getattr(verdict, "status", "PASS")
        if status in ("RUN_INVALID", "BATCH_FAIL"):
            raise GateRejection(
                f"exposure gate rejected request {compiled.request_index}: "
                f"{sorted(set(getattr(verdict, 'reason_codes', ())))}",
                verdict=verdict,
                scope=getattr(verdict, "scope", "request"),
            )

    def _dispatch(self, compiled: CompiledRequest) -> ModelResponse:
        # Persist first, then send the same bytes.  Nothing re-serializes
        # between the record and the transport.
        self.session.append("model.request", compiled.to_json())
        try:
            return self.model.complete(compiled.payload)
        except ProviderError as exc:
            self.session.append(
                "model.failed",
                {
                    "request_index": compiled.request_index,
                    "error": str(exc),
                    "attempts": exc.attempts,
                    "retryable": exc.retryable,
                    "indeterminate": exc.attempts > 1,
                },
            )
            raise

    def _handle_tool_calls(self, tool_calls: list) -> int:
        # Search accounting has four levels, and they must be counted separately
        # or "the model never searched" cannot be distinguished from "the harness
        # never let a search through".  `requested` is incremented here, before any
        # validation, so a malformed or unsupported request still counts as an
        # attempt; `executed` only counts calls that actually ran.
        self.search_requested += sum(
            1 for call in tool_calls
            if (call.get("function", {}) or {}).get("name") == "search"
        )
        handled = 0
        for call in tool_calls:
            function = call.get("function", {})
            name = function.get("name", "")
            arguments = function.get("arguments", {})
            call_id = call.get("id") or f"call-{handled + 1}"

            # The wire format carries arguments as a JSON string; adapters may
            # already have parsed them.  Normalize here so every downstream
            # consumer sees the same shape.
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) if arguments else {}
                except json.JSONDecodeError:
                    arguments = {"__malformed__": arguments}

            if isinstance(arguments, dict) and "__malformed__" in arguments:
                # A malformed argument is a model-side error, returned as an
                # error result so the loop continues as a product would.
                self.session.append(
                    "tool.call",
                    {"call_id": call_id, "tool_name": name, "arguments": {}},
                )
                self.session.append(
                    "tool.raw_result",
                    {"call_id": call_id,
                     "result": {"error": "arguments were not valid JSON"}},
                )
                self.session.append(
                    "tool.visible_result",
                    {"call_id": call_id,
                     "result": {"error": "invalid arguments"}},
                )
                self._record_tool_message(call_id, name, {"error": "invalid arguments"})
                handled += 1
                continue

            if not self.tools.is_enabled(name):
                # Not a harness failure: a real product rejects unknown tools too.
                self.session.append(
                    "tool.call",
                    {"call_id": call_id, "tool_name": name, "arguments": arguments},
                )
                payload = {"error": f"unknown tool {name!r}"}
                self.session.append("tool.raw_result", {"call_id": call_id, "result": payload})
                self.session.append(
                    "tool.visible_result", {"call_id": call_id, "result": payload}
                )
                self._record_tool_message(call_id, name, payload)
                handled += 1
                continue

            self.session.append(
                "tool.call",
                {"call_id": call_id, "tool_name": name,
                 "arguments": json_copy(arguments, "arguments")},
            )
            try:
                result = self.tools.execute(
                    call_id, name, arguments, clock=self.session.clock
                )
            except (ValidationError, ContractViolation):
                raise
            except Exception as exc:  # business-level failure
                self.session.append(
                    "tool.raw_result",
                    {"call_id": call_id, "result": {"error": str(exc)}},
                )
                self.session.append(
                    "tool.visible_result",
                    {"call_id": call_id, "result": {"error": str(exc)}},
                )
                self._record_tool_message(call_id, name, {"error": str(exc)})
                handled += 1
                continue

            self.session.append(
                "tool.raw_result", {"call_id": call_id, "result": result.raw}
            )
            self.session.append(
                "tool.visible_result", {"call_id": call_id, "result": result.visible}
            )
            if name == "search":
                self.search_executed += 1
                if self.tools.last_retrieval is not None:
                    self.session.append(
                        "memory.retrieval", self.tools.last_retrieval.to_json()
                    )
                    # Delivery is a separate fact from execution: a search can run
                    # and return nothing, and the request that follows is where the
                    # evidence either appears or does not.
                    if self.tools.last_retrieval.hits:
                        self.search_evidence_delivered += 1
            self._record_tool_message(call_id, name, result.visible)
            handled += 1
        return handled

    def _record_assistant(self, text: str) -> None:
        seq = self.session.head_seq() + 1
        fragment = self.session.observations.assistant(text, event_seq=seq)
        self.session.append(
            "assistant.message",
            {
                "kind": fragment.kind,
                "text": text,
                "provenance": fragment.provenance.to_json(),
            },
        )

    def _record_assistant_tool_call(self, response) -> None:
        """Record the assistant's tool-requesting turn.

        Normalized to the request shape here so the projection is a re-emission
        rather than a re-derivation: ``arguments`` is stored as the JSON string
        the wire carries, with a missing or unparseable value replaced by an empty
        object rather than dropped, since a call with no arguments is valid and
        distinguishable from a malformed one.
        """
        normalized = []
        for index, call in enumerate(response.tool_calls or []):
            function = call.get("function", {}) or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                arguments_text = arguments
            elif arguments is None:
                arguments_text = "{}"
            else:
                arguments_text = canonical_json(arguments)
            normalized.append(
                {
                    "id": call.get("id") or f"call-{index + 1}",
                    "type": call.get("type", "function"),
                    "function": {
                        "name": function.get("name", ""),
                        "arguments": arguments_text,
                    },
                }
            )
        seq = self.session.head_seq() + 1
        fragment = self.session.observations.assistant_tool_call(
            tool_calls=normalized, event_seq=seq, text=response.content or "",
            reasoning=getattr(response, "reasoning_content", "") or "",
        )
        self.session.append(
            "observation.compiled",
            {
                "kind": fragment.kind,
                "text": fragment.text,
                "tool_calls": normalized,
                "reasoning_content": fragment.meta.get("reasoning_content", ""),
                "provenance": fragment.provenance.to_json(),
            },
        )

    def _record_tool_message(self, call_id: str, tool_name: str, visible: Any) -> None:
        """Append the tool result to model-visible history.

        Separate from the ``tool.visible_result`` control record so that "what
        the tool returned" and "what entered the conversation" stay
        distinguishable in the log.
        """
        text = canonical_json(visible)
        seq = self.session.head_seq() + 1
        fragment = self.session.observations.tool_result(
            text, call_id=call_id, event_seq=seq,
            meta={"call_id": call_id, "tool_name": tool_name},
        )
        self.session.append(
            "observation.compiled",
            {
                "kind": fragment.kind,
                "call_id": call_id,
                "tool_name": tool_name,
                "text": text,
                "provenance": fragment.provenance.to_json(),
            },
        )

    def _fail(self, reason: str, requests: int, tool_calls: int,
              reasons: list, text: str) -> TurnOutcome:
        self.session.append("turn.failed", {"reason": reason})
        outcome = TurnOutcome(
            requests, tool_calls, text, reasons,
            failure={"reason": reason},
        )
        outcome.search = self.search_accounting()
        return outcome
