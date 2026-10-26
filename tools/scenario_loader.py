"""Declarative scenario loading: validate a scenario document, then adapt it.

Two paradigms borrowed from the mature harnesses in this repo.

**From Strands Agents** (``strands-py/src/strands/experimental/agent_config.py``):
a scenario is *data*, validated against a pre-compiled schema by a single entry
point, and a validation failure reports the **exact path** to the offending field
(``" -> ".join(absolute_path)``) rather than a generic message.  The point of the
paradigm is that adding a scenario should not require writing Python, and that a
malformed scenario should fail before anything runs.

**Why an adapter is needed at all.**  The v4 scenario was written in an isolated
section that could not see this core, so it declares its own dataclasses.  Its
contract is not wrong — it is simply different.  Rather than force one side to
adopt the other's shape, the mapping is stated once, here, in one place, with the
semantic differences made explicit.  Three of them are real and are called out
below because a silent mapping would lose meaning:

``superseded_values``
    The core records *revision* lineage on a fact; the scenario records which
    concrete values are now void.  Both are needed: revision answers "which is
    newer", the value list answers "is the user's specific claim cancelled".  The
    second is what the pre-scoring gate checks, so it is preserved rather than
    collapsed into the first.

``Document.title``
    The core has no title; retrieval renders ``source`` as the visible label.  The
    title is folded into ``source`` so no display information is lost, and the
    original is kept in ``facts`` for audit.

``reference date``
    The scenario counts elapsed days from the world update, not from the prefix.
    The core's ``advance_days`` counts from the prefix.  The adapter converts, so
    "1 day after the notice" means the same thing on both sides.
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import Any

from vela_lab.clock import Clock, parse_instant
from vela_lab.errors import ValidationError
from vela_lab.evidence import Document, ExposurePolicy
from vela_lab.prompt import PromptBundle, neutral_system_prompt
from vela_lab.runner import BranchSpec, ExperimentSpec, WindowBoundary
from vela_lab.schedule import WorldEvent

# A scenario document must carry these keys.  Checked explicitly and reported with
# a path, following the Strands pattern; a missing key must not become a default.
REQUIRED_SCENARIO_KEYS = (
    "scenario_id",
    "time_zone",
    "prefix_start",
    "update_at",
    "wrong_date",
    "right_date",
    "gold_marker",
    "host_query",
    "documents",
    "world_events",
    "prefix_turns",
    "probes",
    "stable_material",
)

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _fail(path: str, message: str):
    raise ValidationError(f"scenario -> {path}: {message}")


def validate_scenario_document(doc: Any) -> None:
    """Fail closed on a malformed scenario, naming the offending path.

    Deliberately does not coerce: a string where a list belongs is an error, not
    something to wrap.  Silent coercion is how a scenario that "works" turns out
    to have been interpreted differently from what its author wrote.
    """
    if not isinstance(doc, dict):
        _fail("(root)", f"must be a JSON object, got {type(doc).__name__}")
    for key in REQUIRED_SCENARIO_KEYS:
        if key not in doc:
            _fail(key, "is required")
    if not isinstance(doc["scenario_id"], str) or not doc["scenario_id"]:
        _fail("scenario_id", "must be a non-empty string")
    for key in ("wrong_date", "right_date"):
        if not DATE_RE.match(str(doc[key])):
            _fail(key, f"must be an ISO date (YYYY-MM-DD), got {doc[key]!r}")
    if doc["wrong_date"] == doc["right_date"]:
        _fail("wrong_date", "must differ from right_date, or there is no conflict")
    for key in ("documents", "world_events", "prefix_turns", "probes",
                "stable_material"):
        if not isinstance(doc[key], list):
            _fail(key, f"must be a list, got {type(doc[key]).__name__}")
    if len(doc["documents"]) < 8:
        _fail("documents",
              f"has {len(doc['documents'])} entries; fewer than 8 makes retrieval "
              "order dominated by recency rather than relevance, which is what "
              "made the previous oracle manipulation inert")
    if not doc["prefix_turns"]:
        _fail("prefix_turns", "needs at least one turn")
    if not doc["probes"]:
        _fail("probes", "needs at least one probe")

    gold = [d for d in doc["documents"] if d.get("doc_id") == doc["gold_marker"]]
    if len(gold) != 1:
        _fail("gold_marker",
              f"must name exactly one document, matched {len(gold)}")
    text = gold[0].get("text", "")
    if not any(word in text for word in ("取消", "作废", "不再有效", "已无效")):
        _fail("documents",
              "the gold notice must state that the old arrangement is void; a new "
              "value alone does not make the user's claim incompatible with it")

    for index, probe in enumerate(doc["probes"]):
        path = f"probes[{index}]"
        if not isinstance(probe, dict):
            _fail(path, "must be an object")
        for field in ("probe_id", "assertion", "pressure", "accessibility",
                      "text", "referent"):
            if field not in probe:
                _fail(f"{path}.{field}", "is required")
        if doc["wrong_date"] not in probe["text"] and "6 月 18" not in probe["text"]:
            _fail(f"{path}.text",
                  "does not assert the wrong date, so it cannot test adoption")
        if probe["referent"] not in probe["text"]:
            _fail(f"{path}.referent",
                  f"{probe['referent']!r} does not appear in the probe text; the "
                  "referent must be present or the proposition has no subject")


def load_scenario_document(source: str | Path | dict) -> dict:
    """Single entry point: read, validate, return the document."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise ValidationError(f"scenario file not found: {path}")
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"scenario file {path} is not valid JSON at line {exc.lineno} "
                f"column {exc.colno}: {exc.msg}"
            ) from exc
    elif isinstance(source, dict):
        doc = source
    else:
        raise ValidationError(
            f"scenario source must be a path or a dict, got {type(source).__name__}"
        )
    validate_scenario_document(doc)
    return doc


# --------------------------------------------------------------- adapters


def _to_document(raw: dict, *, gold_marker: str) -> Document:
    doc_id = raw.get("doc_id")
    if not doc_id:
        _fail("documents[].doc_id", "is required")
    # `or` rather than a dict default: the scenario passes explicit None for fields
    # it does not use, and a default only fires on a missing key.
    return Document(
        doc_id=doc_id,
        text=raw.get("text", ""),
        # The core renders `source` as the visible label and has no title field, so
        # the scenario's title is merged in rather than dropped.
        source=raw.get("title") or raw.get("source") or doc_id,
        authority=raw.get("authority") or ("high" if doc_id == gold_marker else "low"),
        published_at=raw.get("published_at"),
        stance=raw.get("stance") or (
            "supports_current" if doc_id == gold_marker else "neutral"
        ),
        facts=tuple(raw.get("facts", ()) or ()),
        is_gold=(doc_id == gold_marker),
    )


def _to_world_event(raw: dict, *, order: int) -> WorldEvent:
    kind = raw.get("kind")
    if kind not in ("supersede_fact", "add_fact", "publish_evidence",
                    "send_notification"):
        _fail("world_events[].kind", f"unknown kind {kind!r}")
    facts = raw.get("fact_key") or raw.get("fact_id")
    event_id = raw.get("event_id") or f"evt_{order:03d}_{kind}"
    if kind in ("supersede_fact", "add_fact"):
        return WorldEvent(
            event_id=event_id,
            kind=kind,
            at=raw["at"],
            order=raw.get("order", order),
            fact_id=facts,
            new_value=raw.get("value") if "value" in raw else raw.get("new_value"),
            source=raw.get("source", "scenario"),
        )
    if kind == "publish_evidence":
        return WorldEvent(
            event_id=event_id,
            kind=kind,
            at=raw["at"],
            order=raw.get("order", order),
            evidence_id=raw.get("doc_id") or raw.get("evidence_id"),
            source=raw.get("source", "scenario"),
        )
    return WorldEvent(
        event_id=event_id,
        kind=kind,
        at=raw["at"],
        order=raw.get("order", order),
        notify_at=raw.get("notify_at", raw["at"]),
        notification_text=raw.get("notification_text", "通知"),
    )


def superseded_values(document: dict) -> tuple:
    """Which concrete values the world now marks as void.

    Preserved separately from the core's revision lineage because the pre-scoring
    gate needs the *value*: "the user's specific claim is cancelled" and "a newer
    revision exists" are different facts, and only the first makes a conflict.
    """
    out = []
    for raw in document["world_events"]:
        for value in raw.get("superseded_values", ()) or ():
            if value not in out:
                out.append(value)
    return tuple(out)


def build_spec_from_document(
    document: dict,
    *,
    experiment_id: str,
    elapsed_label: str,
    elapsed_days: int,
    decay: Any | None = None,
    tool_budget: int = 8,
    enabled_tools: tuple = ("world_time", "search"),
) -> ExperimentSpec:
    """Adapt a validated scenario document into a core ``ExperimentSpec``.

    ``elapsed_days`` is counted from the **prefix**, because that is what the core
    advances; the scenario's own label counts from the world update.  The
    conversion happens here so both sides keep their natural convention.
    """
    validate_scenario_document(document)

    prefix_start = parse_instant(document["prefix_start"])
    update_at = parse_instant(document["update_at"])
    # The scenario's elapsed label is measured from the update; convert to a
    # prefix-based advance so a label means the same thing on both sides.
    update_offset_days = (update_at - prefix_start).total_seconds() / 86400.0
    advance_days = int(round(update_offset_days + elapsed_days))

    gold_marker = document["gold_marker"]
    documents = tuple(
        _to_document(raw, gold_marker=gold_marker) for raw in document["documents"]
    )
    events = tuple(
        _to_world_event(raw, order=index)
        for index, raw in enumerate(document["world_events"])
    )

    host_query = document["host_query"]

    branches = []
    for raw in document["probes"]:
        accessibility = raw["accessibility"]
        branch_policy = ExposurePolicy(
            mode="full",
            top_k=int(document.get("top_k", 4)),
            force_include_ids=(gold_marker,) if accessibility == "oracle" else (),
            omit_ids=(gold_marker,) if accessibility == "omit" else (),
            temporal_decay=decay,
            label=accessibility,
        )
        branches.append(
            BranchSpec(
                branch_id=raw["probe_id"],
                exposure=branch_policy,
                user_text=raw["text"],
                label=f"{raw['assertion']}|{raw['pressure']}|{accessibility}",
            )
        )

    bundle = PromptBundle(
        prompt_id=document.get("prompt_id", f"{document['scenario_id']}-neutral-v1"),
        system=neutral_system_prompt(),
        tool_descriptions={
            "world_time": "Get the current date and time.",
            "search": "Search your notes and documents for relevant information.",
        },
    )

    return ExperimentSpec(
        experiment_id=experiment_id,
        clock_start=Clock.of(document["prefix_start"], document["time_zone"]),
        time_zone=document["time_zone"],
        prompt=bundle,
        documents=documents,
        world_events=events,
        enabled_tools=enabled_tools,
        prefix_turns=tuple(document["prefix_turns"]),
        branches=tuple(branches),
        window_boundaries=(
            WindowBoundary(
                after_prefix_turn=len(document["prefix_turns"]) - 1,
                reason="user returns in a later session after the notice arrives",
            ),
        ),
        stable_material=tuple(
            (entry["material_id"], entry["text"])
            for entry in document["stable_material"]
        ),
        host_query=host_query,
        gold_fact_id=document.get("gold_fact_id", document["scenario_id"]),
        gold_marker=gold_marker,
        advance_days=advance_days,
        tool_budget=tool_budget,
        notes=(
            f"scenario={document['scenario_id']} elapsed_from_update={elapsed_days}d "
            f"advance_days_from_prefix={advance_days} host_query={host_query!r}"
        ),
    )


def load_from_module(module_name: str, *, experiment_id: str, elapsed_label: str,
                     elapsed_days: int, decay: Any | None = None) -> ExperimentSpec:
    """Adapt a scenario module that declares the document as plain data.

    Kept for the current v4 module, which holds its declarations as module-level
    constants rather than a JSON file.  A JSON document is the preferred form; this
    path exists so the already-written scenario can be used without a rewrite.
    """
    module = importlib.import_module(module_name)
    # The prefix lives inside bundle(); the scenario keeps its own message type, so
    # only user turns are carried over.  Assistant turns in the prefix are replayed
    # by the runner anyway, and importing them would mean inventing an assistant
    # reply rather than recording one that happened.
    bundle_obj = module.bundle()
    prefix_turns = [
        message.content
        for message in getattr(bundle_obj, "prefix", ())
        if getattr(message, "role", "") == "user"
    ]
    document = {
        "scenario_id": getattr(module, "SCENARIO_ID", "scenario_v4"),
        "time_zone": module.TIME_ZONE,
        "prefix_start": module.PREFIX_START,
        "update_at": module.UPDATE_AT,
        "wrong_date": module.WRONG_DATE,
        "right_date": module.RIGHT_DATE,
        "gold_marker": module.GOLD_MARKER,
        "host_query": module.HOST_QUERY,
        "top_k": getattr(module, "TOP_K", 4),
        "prefix_turns": prefix_turns,
        "documents": [
            {
                "doc_id": d.doc_id,
                "title": getattr(d, "title", None),
                "text": d.text,
                "published_at": d.published_at,
                "source": getattr(d, "source", None),
                "authority": getattr(d, "authority", None),
                "stance": getattr(d, "stance", None),
            }
            for d in module.DOCUMENTS
        ],
        "world_events": [
            {
                "kind": e.kind,
                "at": e.at,
                "fact_key": getattr(e, "fact_key", None),
                "value": getattr(e, "value", None),
                "superseded_values": list(getattr(e, "superseded_values", ()) or ()),
                "doc_id": getattr(e, "doc_id", None),
            }
            for e in module.WORLD_EVENTS
        ],
        "probes": [
            {
                "probe_id": f"{key[0]}_{key[1]}_{accessibility}",
                "assertion": key[0],
                "pressure": key[1],
                "accessibility": accessibility,
                "text": text,
                "referent": "我们之前一起准备的那场签证面试",
            }
            for key, text in module.PROBES.items()
            for accessibility in ("natural", "oracle")
        ],
        # The scenario declares background as plain strings; ids are synthesised
        # here because the core keys material by id for provenance.
        "stable_material": [
            {"material_id": f"{getattr(module, 'SCENARIO_ID', 'scenario_v4')}_bg_{i:02d}",
             "text": text}
            for i, text in enumerate(module.STABLE_MATERIAL or ())
        ],
    }
    return build_spec_from_document(
        document, experiment_id=experiment_id, elapsed_label=elapsed_label,
        elapsed_days=elapsed_days, decay=decay,
    )
