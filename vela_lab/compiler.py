"""The single request outlet.

Contract §4 and §10 require that the harness can prove what the model was
actually sent.  That is only possible if there is exactly one place where a
request is assembled, and if that place is auditable.

Two structural properties:

**It accepts only ``VisibleFragment``.**  Not strings, not dicts, not message
objects.  Since fragments can only be minted by ``ObservationDomain``, the
compiler is type-level incapable of receiving gold, judge labels, or world
truth.  This is the mechanism, not a convention.

**It returns a frozen record with a payload digest.**  The loop persists the
record *before* dispatch and sends the recorded payload — not a re-serialized
equivalent — so "recorded" and "sent" cannot drift.

What this proves and what it does not:

* proves: the recorded bytes are the bytes handed to the transport, and every
  fragment traces to a registered template or a logged observation.
* does not prove: what the hosted model did internally with those bytes.  Do not
  claim otherwise.  Also do not claim the absence of *semantic* leakage from a
  clean keyword scan — the scan is a backstop, the gate is the guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .errors import ContractViolation, ValidationError
from .domains import ObservationDomain, VisibleFragment, assert_no_forbidden_terms
from .jsonutil import canonical_json, digest, json_copy
from .prompt import PromptBundle

# Roles accepted by the OpenAI-compatible chat shape.
_ROLE_OF = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool_result": "tool",
    "memory_note": "user",       # memory notes fold into the user turn
    "time_context": "user",
    "stable_material": "user",   # folded into the first user turn of the window
    "assistant_tool_call": "assistant",  # the model's tool-requesting turn
}


@dataclass(frozen=True)
class CompiledRequest:
    """An immutable request record plus everything audit needs to re-verify it."""

    request_index: int
    prompt_id: str
    prompt_digest: str
    messages: list
    tools: list
    payload: dict
    payload_digest: str
    fragment_provenance: list = field(default_factory=list)
    context_manifest: list = field(default_factory=list)
    applied_param_keys: list = field(default_factory=list)
    bound_at_seq: int = 0

    def to_json(self) -> dict:
        return {
            "request_index": self.request_index,
            "prompt_id": self.prompt_id,
            "prompt_digest": self.prompt_digest,
            "messages": self.messages,
            "tools": self.tools,
            "payload": self.payload,
            "payload_digest": self.payload_digest,
            "fragment_provenance": self.fragment_provenance,
            "context_manifest": self.context_manifest,
            "applied_param_keys": self.applied_param_keys,
            "bound_at_seq": self.bound_at_seq,
        }

    @staticmethod
    def from_json(payload: dict) -> "CompiledRequest":
        required = ("request_index", "prompt_id", "prompt_digest", "messages",
                    "tools", "payload", "payload_digest")
        missing = [key for key in required if key not in payload]
        if missing:
            from .errors import ValidationError

            raise ValidationError(f"compiled request is missing {missing}")
        return CompiledRequest(
            request_index=payload["request_index"],
            prompt_id=payload["prompt_id"],
            prompt_digest=payload["prompt_digest"],
            messages=json_copy(payload["messages"], "messages"),
            tools=json_copy(payload["tools"], "tools"),
            payload=json_copy(payload["payload"], "payload"),
            payload_digest=payload["payload_digest"],
            fragment_provenance=json_copy(
                payload.get("fragment_provenance", []), "provenance"
            ),
            context_manifest=json_copy(
                payload.get("context_manifest", []), "context manifest"
            ),
            applied_param_keys=list(payload.get("applied_param_keys", [])),
            bound_at_seq=payload.get("bound_at_seq", 0),
        )


class RequestCompiler:
    """Turns observations into a request.  The only path to a provider."""

    def __init__(self, bundle: PromptBundle, *, tool_schemas: Sequence[dict]):
        self.bundle = bundle
        self.tool_schemas = [json_copy(s, "tool schema") for s in tool_schemas]
        _validate_tool_schemas(self.tool_schemas)

    # ------------------------------------------------------------- compiling

    def compile(
        self,
        observations: ObservationDomain,
        *,
        request_index: int,
        bound_at_seq: int = 0,
        extra_params: dict | None = None,
    ) -> CompiledRequest:
        fragments = list(observations.fragments())
        if not fragments:
            raise ContractViolation("cannot compile a request with no observations")

        # Backstop: vocabulary scan over everything about to become visible.
        assert_no_forbidden_terms(fragments)

        messages = self._messages(fragments)
        payload = {
            "messages": messages,
            "tools": self.tool_schemas,
        }
        params = extra_params or {}
        for key, value in params.items():
            if key in ("messages", "tools"):
                raise ContractViolation(
                    f"extra_params may not override {key!r}; that would bypass the "
                    "registered prompt and tool bundle"
                )
            payload[key] = json_copy(value, f"param[{key}]")

        payload_digest = digest(payload)

        # Context-construction record.  Answers "which material entered *this*
        # request, from where, and what was left out" without making a reader
        # reconstruct it from the message text.  A per-branch flag such as
        # "evidence was exposed" is not enough: the same branch can legitimately
        # show the evidence on one call and not the next, and collapsing that
        # distinction is how "saw it but ignored it" gets inferred from a call
        # that never showed it.
        manifest = []
        for position, fragment in enumerate(fragments):
            entry = {
                "position": position,
                "kind": fragment.kind,
                "source": fragment.provenance.source,
                "ref": fragment.provenance.ref,
                "chars": len(fragment.text),
                "content_digest": digest(fragment.text),
                # Window membership is what makes a cross-window leak detectable.
                # A content digest alone cannot do it: two independent fragments
                # with identical text (the same search returning the same
                # documents in two windows) collide, so a digest-only check
                # cannot separate "the dropped fragment came back" from "a new
                # fragment happens to have identical content".
                "window": fragment.window,
                "frozen": fragment.frozen,
            }
            if fragment.meta.get("doc_id"):
                entry["doc_id"] = fragment.meta["doc_id"]
                entry["is_gold"] = bool(fragment.meta.get("is_gold"))
            if fragment.meta.get("call_id"):
                entry["call_id"] = fragment.meta["call_id"]
            manifest.append(entry)

        return CompiledRequest(
            request_index=request_index,
            prompt_id=self.bundle.prompt_id,
            prompt_digest=self.bundle.digest(),
            messages=messages,
            tools=self.tool_schemas,
            payload=payload,
            payload_digest=payload_digest,
            fragment_provenance=[f.provenance.to_json() for f in fragments],
            context_manifest=manifest,
            applied_param_keys=sorted(params),
            bound_at_seq=bound_at_seq,
        )

    # -------------------------------------------------------------- internals

    def _messages(self, fragments: Sequence[VisibleFragment]) -> list:
        """Project fragments into a chat request.

        Memory notes and the time line are folded into the user turn they belong
        to, rather than emitted as separate messages.  Two reasons:

        * it matches how a real assistant receives retrieved context — inside the
          current turn, not as a synthetic message;
        * and it keeps the message sequence protocol-shaped.  Emitting a note as
          its own message after a tool result would put a user message where the
          protocol expects the tool result, which both breaks tool continuations
          and misrepresents who said what.

        A note is attached to the user turn that is open when it appears.  Notes
        arriving before any user turn are held and attached to the first one; a
        note after a tool result still belongs to the same turn, so it is merged
        into that turn's user message rather than starting a new one.
        """
        messages = [{"role": "system", "content": self.bundle.system}]
        turn_index: int | None = None      # index in ``messages`` of the open user turn
        pending_time: list[str] = []       # time lines with no user turn yet
        pending_notes: list[str] = []      # memory notes with no user turn yet

        for fragment in fragments:
            if fragment.kind == "system":
                continue

            if fragment.kind == "time_context":
                if turn_index is None:
                    pending_time.append(fragment.text)
                else:
                    messages[turn_index]["content"] = self._merge(
                        messages[turn_index]["content"], fragment.text
                    )
                continue
            if fragment.kind in ("memory_note", "stable_material"):
                if turn_index is None:
                    pending_notes.append(fragment.text)
                else:
                    messages[turn_index]["content"] = self._merge(
                        messages[turn_index]["content"],
                        self._wrap_notes([fragment.text]),
                    )
                continue

            role = _ROLE_OF.get(fragment.kind)
            if role is None:
                raise ContractViolation(f"cannot project fragment kind {fragment.kind!r}")

            if role == "user":
                preamble = pending_time + (
                    [self._wrap_notes(pending_notes)] if pending_notes else []
                )
                content = "\n\n".join(preamble + [fragment.text]) if preamble else fragment.text
                pending_time, pending_notes = [], []
                messages.append({"role": "user", "content": content})
                turn_index = len(messages) - 1
            elif fragment.kind == "assistant_tool_call":
                # Re-emit the model's own tool-requesting turn.  Omitting it would
                # leave the following tool result answering nothing, which strict
                # providers reject and tolerant ones accept while the model sees a
                # result appear from nowhere.  Arguments go back as the JSON string
                # the wire expects, since that is what was recorded.
                calls = []
                for call in fragment.meta.get("tool_calls", []):
                    function = call.get("function", {})
                    arguments = function.get("arguments")
                    if not isinstance(arguments, str):
                        arguments = canonical_json(arguments if arguments is not None else {})
                    calls.append(
                        {
                            "id": call.get("id", ""),
                            "type": call.get("type", "function"),
                            "function": {
                                "name": function.get("name", ""),
                                "arguments": arguments,
                            },
                        }
                    )
                message = {
                    "role": "assistant",
                    "content": fragment.text or "",
                    "tool_calls": calls,
                }
                # Replay the provider's hidden reasoning when one was returned.
                #
                # DeepSeek's thinking mode rejects a continuation whose assistant
                # message omits it (HTTP 400, "The `reasoning_content` in the
                # thinking mode must be passed back to the API"), so leaving it out
                # turns a tool round into a provider error for one route while
                # every other route works.  Echoing what the provider itself sent
                # changes nothing the model is being told; it restores the bytes
                # that route expects to see.
                reasoning = fragment.meta.get("reasoning_content") or ""
                if reasoning:
                    message["reasoning_content"] = reasoning
                messages.append(message)
                # A tool-call turn does not close the user turn: the results that
                # follow still belong to it, so notes keep attaching there.
            elif role == "tool":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": fragment.meta.get("call_id", ""),
                        "name": fragment.meta.get("tool_name", ""),
                        "content": fragment.text,
                    }
                )
                # A tool result closes the model's opportunity to speak, but the
                # user turn is still open: notes arriving now belong to it.
            else:
                messages.append({"role": role, "content": fragment.text})
                turn_index = None

        if pending_time or pending_notes:
            # Nothing else to attach to: emit the time line and notes as the
            # final user turn so no observation is silently dropped.
            parts = list(pending_time)
            if pending_notes:
                parts.append(self._wrap_notes(pending_notes))
            messages.append({"role": "user", "content": "\n\n".join(parts)})
        return messages

    def _wrap_notes(self, notes: list) -> str:
        """Render notes through the registered wrapper.

        Using the bundle's wrapper matters for more than tidiness: the wrapper is
        part of the frozen prompt identity, and it is what delineates quoted
        material from the user's own words.  Without a boundary the model can
        read retrieved evidence as something the user said, which changes source
        attribution — a different variable from accessibility.
        """
        entries = "\n".join(f"- {line}" for line in notes)
        return self.bundle.memory_wrapper.format(entries=entries)

    def _merge(self, content: str, addition: str) -> str:
        """Attach material to an existing user turn without reordering it."""
        if not content:
            return addition
        return f"{content}\n\n{addition}"


def _validate_tool_schemas(schemas: Sequence[dict]) -> None:
    seen = set()
    for schema in schemas:
        if not isinstance(schema, dict):
            raise ValidationError("tool schema must be a mapping")
        function = schema.get("function")
        if not isinstance(function, dict):
            raise ValidationError("tool schema requires a 'function' object")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValidationError("tool schema requires a non-empty function name")
        if name in seen:
            raise ValidationError(f"duplicate tool name {name!r}")
        seen.add(name)
        if not isinstance(function.get("description"), str):
            raise ValidationError(f"tool {name!r} requires a description")
        params = function.get("parameters")
        if not isinstance(params, dict) or params.get("type") != "object":
            raise ValidationError(
                f"tool {name!r} parameters must be a JSON-schema object"
            )
    for forbidden in ("gold", "oracle", "condition", "branch", "harness"):
        for schema in schemas:
            name = schema["function"]["name"].lower()
            if forbidden in name:
                raise ContractViolation(
                    f"tool name {name!r} contains experiment vocabulary {forbidden!r}; "
                    "tool names reach the model directly"
                )


def verify_compiled_request(record: CompiledRequest) -> list:
    """Re-verify a recorded request offline.  Returns findings, empty if clean."""
    findings = []
    recomputed = digest(record.payload)
    if recomputed != record.payload_digest:
        findings.append(
            {
                "check": "request_payload_digest",
                "detail": f"payload digest {recomputed[:12]} != recorded "
                          f"{record.payload_digest[:12]}",
            }
        )
    if record.messages != record.payload.get("messages"):
        findings.append(
            {
                "check": "request_messages_consistent",
                "detail": "top-level messages differ from the payload messages",
            }
        )
    if record.tools != record.payload.get("tools"):
        findings.append(
            {
                "check": "request_tools_consistent",
                "detail": "top-level tools differ from the payload tools",
            }
        )
    for entry in record.fragment_provenance:
        if not entry.get("source") or not entry.get("ref"):
            findings.append(
                {
                    "check": "fragment_provenance",
                    "detail": f"fragment without traceable origin: {entry}",
                }
            )
    return findings
