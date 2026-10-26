"""Fail-closed offline audit.

Contract §10 and §13: audit is what makes a claim about the model's behaviour
possible at all.  It runs offline, re-derives everything from the log, and
refuses to pass when it cannot confirm a guarantee.  "Cannot confirm" is a
failure, not a warning — the alternative is a run that looks clean because the
checker gave up.

Checks fall into four groups:

**Integrity.** Hash chain, ordinal contiguity, no unknown event types, and the
live state matching a fresh fold.

**Domain gate.** Judge-domain values must not appear anywhere in a recorded
request.  This is checked against recorded *requests*, not intentions — the
question is what was sent.

**Exposure chain.** For each retrieval, verify the recorded arrows: was gold
eligible, did it enter the candidate set, was it selected, and did it reach a
recorded request.  This is the check that the old harness could not perform and
whose absence allowed its central conclusion to rest on a mis-typed comparison.

**Naturalism.** No experiment vocabulary in model-visible content, no
condition labels leaking through call ids or tool names, and no host-clock
dependency.

The audit deliberately does not decide whether a *finding* is interesting.  It
reports what happened.  Interpretation lives in ``score``.
"""

from __future__ import annotations

from typing import Any, Sequence

from .clock import parse_instant
from .compiler import CompiledRequest, verify_compiled_request
from .domains import FORBIDDEN_TERMS
from .errors import AuditFailure, IntegrityError, ValidationError
from .jsonutil import canonical_json
from .log import verify_chain
from .reducer import KIND_TO_ROLE, VALID_EVENT_TYPES


def audit_events(events: list, *, extra_terms: Sequence[str] = (),
                 judge_values: Sequence[Any] = ()) -> list:
    """Audit a raw event list.  Returns findings; empty means clean."""
    findings: list[dict] = []
    if not events:
        return [{"check": "non_empty", "detail": "the log holds no events"}]

    try:
        verify_chain(events)
    except IntegrityError as exc:
        return [{"check": "hash_chain", "detail": str(exc)}]

    if events[0]["type"] != "session.started":
        findings.append(
            {"check": "genesis", "detail": "the first event is not session.started"}
        )

    for event in events:
        if event["type"] not in VALID_EVENT_TYPES:
            findings.append(
                {
                    "check": "known_event_type",
                    "detail": f"seq {event['seq']}: unknown type {event['type']!r}",
                }
            )

    findings.extend(_audit_tool_provenance(events))
    findings.extend(_audit_requests(events, extra_terms, judge_values))
    findings.extend(_audit_exposure_chain(events))
    findings.extend(_audit_control_plane(events))
    findings.extend(_audit_failures(events))
    findings.extend(_audit_window_integrity(events))
    findings.extend(_audit_raw_manifest_isolation(events))
    findings.extend(_audit_call_pairing(events))
    findings.extend(_audit_indeterminate(events))
    return findings


# --------------------------------------------------------------- integrity


def _audit_tool_provenance(events: list) -> list:
    """Every tool result must be preceded by its call, and used if shown."""
    findings = []
    calls: dict[str, int] = {}
    raw: set[str] = set()
    visible: set[str] = set()
    for event in events:
        seq = event["seq"]
        data = event["data"]
        if event["type"] == "tool.call":
            call_id = data.get("call_id")
            if call_id in calls:
                findings.append(
                    {
                        "check": "tool_call_unique",
                        "detail": f"seq {seq}: duplicate call id {call_id!r}",
                    }
                )
            calls[call_id] = seq
        elif event["type"] == "tool.raw_result":
            call_id = data.get("call_id")
            if call_id not in calls:
                findings.append(
                    {
                        "check": "tool_raw_after_call",
                        "detail": f"seq {seq}: raw result for {call_id!r} without a call",
                    }
                )
            raw.add(call_id)
        elif event["type"] == "tool.visible_result":
            call_id = data.get("call_id")
            if call_id not in raw:
                findings.append(
                    {
                        "check": "tool_visible_after_raw",
                        "detail": f"seq {seq}: visible result for {call_id!r} "
                                  "without a raw value",
                    }
                )
            visible.add(call_id)
        elif event["type"] == "observation.compiled" and data.get("kind") == "tool_result":
            call_id = data.get("call_id")
            if call_id not in visible:
                findings.append(
                    {
                        "check": "tool_message_after_visible",
                        "detail": f"seq {seq}: tool message for {call_id!r} "
                                  "without a visible result",
                    }
                )
    unresolved = set(calls) - visible
    if unresolved:
        findings.append(
            {
                "check": "tool_calls_resolved",
                "detail": f"tool calls without a visible result: {sorted(unresolved)}",
            }
        )
    return findings


# ------------------------------------------------------------- domain gate


def _harness_authored(messages: Sequence[dict]) -> list:
    """The visible messages the harness authored or selected, not the model.

    Why provenance decides this.  The vocabulary check exists to catch the harness
    leaking experiment words into what the model reads — an "oracle" left in a
    template, "gold" in a retrieved document.  A *model-authored* message is not a
    leak no matter what it says: the route writing "dental treatment follow-up"
    about a dental appointment, or "conditional" while hedging, or "golden week" for
    a holiday, is using ordinary English.

    That distinction became load-bearing when the assistant's hidden reasoning began
    being replayed into later requests (``reasoning_content``, which one route
    requires back on a tool continuation).  Before that, assistant text was the
    model's final answer and mostly short; now it is a long transcript of the
    model's own thinking, and scanning it produced false positives that invalidated
    eleven branches in one batch.  The audit was reporting "the model used a normal
    word" as "the harness leaked".

    The judge-value check below still scans the *whole* payload, which is correct:
    a gold marker appearing anywhere in what is sent is a leak regardless of who
    wrote it.
    """
    return [
        message for message in messages
        if message.get("role") != "assistant"
    ]


def _audit_requests(events: list, extra_terms: Sequence[str],
                    judge_values: Sequence[Any]) -> list:
    """Visible content must be free of experiment vocabulary, and gold absent.

    Two checks with deliberately different scopes: the vocabulary scan reads only
    harness-authored blocks, while the gold scan reads the entire payload.  See
    ``_harness_authored`` for why the first is narrower.
    """
    findings = []
    terms = [t.lower() for t in list(FORBIDDEN_TERMS) + list(extra_terms)]
    needles = [canonical_json(v) for v in judge_values if v is not None]

    for event in events:
        if event["type"] != "model.request":
            continue
        data = event["data"]
        blob = canonical_json(
            {"messages": _harness_authored(data.get("messages", []))}
        ).lower()
        for term in terms:
            if term in blob:
                findings.append(
                    {
                        "check": "visible_vocabulary",
                        "detail": f"seq {event['seq']}: request contains {term!r}",
                    }
                )
        payload_blob = canonical_json(data.get("payload", {}))
        for needle in needles:
            if needle and needle in payload_blob:
                findings.append(
                    {
                        "check": "judge_value_absent_from_request",
                        "detail": f"seq {event['seq']}: request payload contains a "
                                  "judge-domain value",
                    }
                )
        try:
            record = CompiledRequest.from_json(data)
        except ValidationError as exc:
            findings.append(
                {
                    "check": "request_record_well_formed",
                    "detail": f"seq {event['seq']}: unparseable request record ({exc})",
                }
            )
            continue
        findings.extend(
            {
                "check": finding["check"],
                "detail": f"seq {event['seq']}: {finding['detail']}",
            }
            for finding in verify_compiled_request(record)
        )
    return findings

# ----------------------------------------------------------- exposure chain


def _audit_window_integrity(events: list) -> list:
    """A window switch must actually remove earlier-window content.

    This is the check that makes a cross-window accessibility manipulation
    meaningful.  If the established fact is still sitting in the raw history
    after the switch, then the manipulation changed nothing and any conclusion
    about accessibility is unsupported — which is precisely the defect the
    project already paid for once.

    Three things are verified:

    * the manifest enumerates retained and dropped fragments, each with a
      digest, so the switch is auditable rather than asserted;
    * nothing dropped earlier reappears in a later request;
    * every request after a switch belongs to the window then open.
    """
    findings = []

    # A fragment cannot be listed as both retained and dropped.
    for event in events:
        if event["type"] != "context.rebased":
            continue
        data = event["data"]
        dropped = {
            entry.get("content_digest") for entry in data.get("dropped", [])
        }
        for entry in data.get("retained", []):
            if entry.get("content_digest") in dropped:
                findings.append(
                    {
                        "check": "rebased_manifest_consistent",
                        "detail": f"seq {event['seq']}: fragment {entry.get('ref')!r} "
                                  "is listed as both retained and dropped",
                    }
                )

    # Nothing from a closed window may appear in a request made after the
    # switch.  Checked by window membership rather than by content digest: two
    # independent fragments can have identical text, so a digest comparison
    # would report a leak whenever the same search ran twice.
    current_window = 0
    for event in events:
        etype, data, seq = event["type"], event["data"], event["seq"]
        if etype == "session.opened":
            current_window = data.get("window", current_window)
            continue
        if etype == "context.rebased":
            # The rebase opens the window it names.
            current_window = data.get("window", current_window + 1)
            continue
        if etype != "model.request":
            continue
        for entry in data.get("context_manifest", []):
            if entry.get("frozen"):
                # Registered background is present by design and is not subject
                # to the switch.
                continue
            window = entry.get("window")
            if window is None:
                findings.append(
                    {
                        "check": "manifest_entry_names_window",
                        "detail": f"seq {seq}: manifest entry {entry.get('ref')!r} has "
                                  "no window, so cross-window leakage cannot be ruled out",
                    }
                )
            elif window < current_window:
                findings.append(
                    {
                        "check": "closed_window_content_not_redelivered",
                        "detail": f"seq {seq}: fragment {entry.get('ref')!r} belongs to "
                                  f"window {window} but is delivered in window "
                                  f"{current_window}; a closed window's content must not "
                                  "reach a later request",
                    }
                )
    return findings


def _audit_raw_manifest_isolation(events: list) -> list:
    """Internal identifiers and gold labels must never reach the sent body.

    The context manifest records internal ``doc_id``s and which document is
    gold, because audit and scoring need that.  It is control-plane data.  If it
    ever appeared inside the payload handed to the provider, the model would
    receive exactly the labels this harness exists to withhold — and the
    manifest is the most likely place for that to happen by accident, since it
    is built in the same function that builds the request.
    """
    findings = []
    for event in events:
        if event["type"] != "model.request":
            continue
        data = event["data"]
        payload = data.get("payload")
        if not isinstance(payload, dict):
            continue
        # 1. The transmitted body must contain only the allowed top-level keys.
        #    A structural check beats substring matching: it catches a whole
        #    control-plane object being attached, which is the realistic mistake.
        allowed = {"messages", "tools"} | set(data.get("applied_param_keys") or [])
        extra = sorted(set(payload) - allowed)
        if extra:
            findings.append(
                {
                    "check": "payload_keys_allowed",
                    "detail": f"seq {event['seq']}: transmitted payload carries "
                              f"unexpected keys {extra}",
                }
            )

        # 2. Internal marker keys must not appear as JSON keys anywhere in the
        #    body, at any nesting depth — including inside a string that itself
        #    contains serialized JSON, which is how an internal envelope would
        #    realistically arrive (a tool result carrying its own metadata).
        #
        #    Checked as keys rather than as free text: a user message that happens
        #    to contain the word "gold" is not a leak, and treating it as one
        #    would make the check useless through false positives.
        for key in _find_internal_keys(payload):
            findings.append(
                {
                    "check": "internal_key_not_in_payload",
                    "detail": f"seq {event['seq']}: internal key {key!r} appears in "
                              "the transmitted payload",
                }
            )
    return findings


INTERNAL_KEYS = frozenset(
    {
        "is_gold",
        "doc_id",
        "context_manifest",
        "fragment_provenance",
        "content_digest",
        "gold",
        "oracle_forced",
        "eligible_ids",
        "candidates",
        "hidden_ids",
    }
)


def _find_internal_keys(node, found: set | None = None, depth: int = 0) -> set:
    """Walk a JSON value for internal keys, descending into nested JSON strings.

    A tool result's content is often a string containing JSON.  Scanning only
    the outer structure would miss an internal envelope nested one level down,
    which is exactly where it would realistically appear.
    """
    if found is None:
        found = set()
    if depth > 8:
        return found
    if isinstance(node, dict):
        for key, value in node.items():
            if key in INTERNAL_KEYS:
                found.add(key)
            _find_internal_keys(value, found, depth + 1)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _find_internal_keys(item, found, depth + 1)
    elif isinstance(node, str):
        stripped = node.strip()
        if stripped[:1] in ("{", "["):
            import json as _json

            try:
                parsed = _json.loads(stripped)
            except Exception:
                return found
            _find_internal_keys(parsed, found, depth + 1)
    return found


def _audit_exposure_chain(events: list) -> list:
    """Verify the arrows from 'gold exists' to 'gold reached the request'."""
    findings = []
    retrievals = [e for e in events if e["type"] == "memory.retrieval"]
    requests = [e for e in events if e["type"] == "model.request"]

    for retrieval in retrievals:
        data = retrieval["data"]
        hits = data.get("hits", [])
        gold_hits = [h for h in hits if h.get("is_gold")]
        eligible = set(data.get("eligible_ids", []))
        gold_eligible = any(
            h.get("is_gold") for h in data.get("candidates", [])
        )
        if data.get("oracle_forced"):
            if not gold_hits:
                findings.append(
                    {
                        "check": "oracle_delivered_gold",
                        "detail": f"seq {retrieval['seq']}: oracle forced "
                                  f"{data['oracle_forced']} but no gold hit is present",
                    }
                )
        # An oracle retrieval that changed the result count would confound
        # accessibility with context volume.
        if data.get("oracle_forced"):
            if len(hits) != min(data.get("top_k", len(hits)), len(eligible)):
                findings.append(
                    {
                        "check": "oracle_count_invariant",
                        "detail": f"seq {retrieval['seq']}: oracle changed the "
                                  "number of visible documents",
                    }
                )
        # Every retrieval must record its corpus identity, or it cannot be
        # reproduced offline.
        if not data.get("corpus_digest") or not data.get("retriever_id"):
            findings.append(
                {
                    "check": "retrieval_identifiable",
                    "detail": f"seq {retrieval['seq']}: retrieval lacks a corpus "
                              "digest or retriever id",
                }
            )
        if not gold_eligible and gold_hits:
            findings.append(
                {
                    "check": "gold_eligible_if_returned",
                    "detail": f"seq {retrieval['seq']}: gold returned but absent "
                              "from the candidate set",
                }
            )

    # Did a selected document ever reach a request?  Compared against the tool
    # message *content* rather than a re-serialization of the whole message
    # list, because embedding a JSON payload inside a JSON envelope escapes its
    # quotes and would make a naive substring test fail on a document that was
    # in fact delivered.
    delivered_texts = []
    for request in requests:
        for message in request["data"].get("messages", []):
            content = message.get("content")
            if isinstance(content, str):
                delivered_texts.append(content)
    for retrieval in retrievals:
        for hit in retrieval["data"].get("hits", []):
            rendered = hit.get("visible_text")
            if not rendered:
                findings.append(
                    {
                        "check": "hit_records_rendered_text",
                        "detail": f"seq {retrieval['seq']}: hit {hit['doc_id']!r} has no "
                                  "recorded rendering, so delivery cannot be verified",
                    }
                )
                continue
            if not any(rendered in text for text in delivered_texts):
                findings.append(
                    {
                        "check": "retrieved_then_delivered",
                        "detail": f"seq {retrieval['seq']}: document {hit['doc_id']!r} was "
                                  "selected but its rendered text never appears in a "
                                  "recorded request",
                    }
                )
    return findings


# ---------------------------------------------------------- control plane


def _audit_control_plane(events: list) -> list:
    """Control events must never become model-visible text."""
    findings = []
    for event in events:
        if event["type"] not in ("clock.advance", "world.fact_set",
                                 "world.fact_superseded", "world.event_applied",
                                 "prompt.bound", "checkpoint.created",
                                 "branch.forked", "tool.raw_result",
                                 "tool.visible_result", "memory.retrieval",
                                 "model.request", "model.response", "model.failed",
                                 "turn.started", "turn.completed", "turn.failed",
                                 "session.started", "fault.applied"):
            continue
        # A control event must not be a history appender.
        if event["type"] in KIND_TO_ROLE:
            findings.append(
                {
                    "check": "control_not_history",
                    "detail": f"seq {event['seq']}: control event {event['type']!r} "
                              "reached model history",
                }
            )
    return findings


def _audit_failures(events: list) -> list:
    """A failed turn must terminate the run, not be papered over."""
    findings = []
    failed_at = None
    for event in events:
        if event["type"] == "turn.failed":
            failed_at = event["seq"]
        elif failed_at is not None and event["type"] in ("turn.started",
                                                         "model.request"):
            findings.append(
                {
                    "check": "failure_is_terminal",
                    "detail": f"seq {event['seq']}: activity after the failure at "
                              f"seq {failed_at}",
                }
            )
    return findings


def _audit_call_pairing(events: list) -> list:
    """Every request must be answered by exactly one response or failure.

    A request with neither is the worst case for research: the log shows the
    model was asked something, and nothing says what came back.  That happens
    when a process dies mid-call, or when a retry is treated as success.
    """
    findings = []
    requests: dict[int, int] = {}
    answered: dict[int, list] = {}
    for event in events:
        data = event["data"]
        if event["type"] == "model.request":
            requests[data.get("request_index")] = event["seq"]
        elif event["type"] == "model.response":
            answered.setdefault(data.get("request_index"), []).append(
                ("response", event["seq"])
            )
        elif event["type"] == "model.failed":
            answered.setdefault(data.get("request_index"), []).append(
                ("failed", event["seq"])
            )

    for index, seq in sorted(requests.items()):
        outcomes = answered.get(index, [])
        if not outcomes:
            findings.append(
                {
                    "check": "request_has_outcome",
                    "detail": f"request_index {index} (seq {seq}) has neither a "
                              "response nor a recorded failure",
                }
            )
        elif len(outcomes) > 1:
            findings.append(
                {
                    "check": "request_answered_once",
                    "detail": f"request_index {index} has {len(outcomes)} outcomes "
                              f"{outcomes}; a request must be answered once",
                }
            )

    for index, outcomes in answered.items():
        if index not in requests:
            findings.append(
                {
                    "check": "outcome_has_request",
                    "detail": f"outcome(s) {outcomes} reference request_index "
                              f"{index}, which was never recorded as sent",
                }
            )
    return findings


def _audit_indeterminate(events: list) -> list:
    """A call whose outcome is unknown must be visible as such.

    If a retry succeeded after a first attempt may have reached the provider,
    the response is not a clean single observation.  Recording it without
    flagging would let duplicate side effects pass as one model behaviour.
    """
    findings = []
    for event in events:
        if event["type"] != "model.response":
            continue
        response = event["data"].get("response", {})
        attempts = response.get("attempts", 1)
        if attempts > 1:
            findings.append(
                {
                    "check": "indeterminate_call_flagged",
                    "detail": f"seq {event['seq']}: response after {attempts} attempts; "
                              "the earlier attempt's outcome is unknown",
                }
            )
    return findings


# -------------------------------------------------------------- host clock


def audit_no_host_clock(events: list, *, host_window: tuple) -> list:
    """Ensure no virtual timestamp looks like a host clock reading.

    ``host_window`` is the (start, end) wall-clock window of the run.  A virtual
    instant inside it is suspicious: it suggests something read the host clock
    instead of the session's clock.
    """
    findings = []
    start, end = host_window
    for event in events:
        stamp = event.get("virtual_time")
        if not stamp:
            continue
        try:
            moment = parse_instant(stamp)
        except Exception:
            findings.append(
                {
                    "check": "virtual_time_parseable",
                    "detail": f"seq {event['seq']}: unparseable virtual_time {stamp!r}",
                }
            )
            continue
        if start <= moment <= end:
            findings.append(
                {
                    "check": "virtual_time_not_host_clock",
                    "detail": f"seq {event['seq']}: virtual time {stamp} falls inside "
                              "the host wall-clock window of the run",
                }
            )
    return findings


def audit_or_raise(events: list, **kwargs) -> None:
    findings = audit_events(events, **kwargs)
    if findings:
        raise AuditFailure(findings)
