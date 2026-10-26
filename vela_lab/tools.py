"""Read-only tools with an explicit raw/visible split.

Every tool call produces two values: ``raw``, what the world actually returned,
and ``visible``, what the model is shown.  Both are logged and cross-checked.
The split is what makes faults and visibility conditions expressible without
touching the prompt: a condition changes the projection, never the instruction,
so the model has no textual hint that anything was manipulated.

Contract §8: read-only only.  A write tool would make "the tool returned X" and
"the world now contains X" two different claims, and the second one would need
its own audit.  Out of scope for v0.1.

Tool names, descriptions and result payloads must read like an ordinary
product.  ``search``, not ``controlled_search``; "Get the current date and
time", not "Get the current virtual environment time".  The naturalism check in
audit scans for the vocabulary, but the naming discipline comes first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .clock import Clock, format_human
from .errors import ContractViolation, ValidationError
from .evidence import EvidenceCorpus, ExposurePolicy, RetrievalRecord
from .jsonutil import canonical_json

TOOL_NAMES = ("world_time", "search", "calendar_events", "weather", "official_registry")


@dataclass(frozen=True)
class ToolResult:
    """A tool's paired raw and visible values."""

    call_id: str
    tool_name: str
    raw: Any
    visible: Any
    error: dict | None = None

    def visible_text(self) -> str:
        if self.error:
            return canonical_json(self.error)
        return _render(self.visible)

    def to_json(self) -> dict:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "raw": self.raw,
            "visible": self.visible,
            "error": self.error,
        }


def _render(value: Any) -> str:
    """Render a tool's visible payload the way a product would: JSON, compact."""
    return canonical_json(value)


class ToolBelt:
    """The registered tool set for a run.

    ``enabled`` is branch-level but must be identical across branches of a
    manipulation — narrowing the tool set changes what the model can do at all,
    which is a different variable from what it can see (contract §8).
    """

    def __init__(
        self,
        *,
        enabled: Sequence[str],
        corpus: EvidenceCorpus,
        world: Any,
        exposure: ExposurePolicy,
        tool_descriptions: dict | None = None,
        faults: dict | None = None,
    ):
        unknown = set(enabled) - set(TOOL_NAMES)
        if unknown:
            raise ValidationError(f"unknown tools requested: {sorted(unknown)}")
        self.enabled = tuple(enabled)
        self.corpus = corpus
        self.world = world
        self.exposure = exposure
        self.descriptions = dict(tool_descriptions or {})
        self.faults = dict(faults or {})
        self._applied_faults: set[str] = set()
        self.last_retrieval: RetrievalRecord | None = None

    # -------------------------------------------------------------- schemas

    def schemas(self) -> list:
        """OpenAI-compatible tool schemas, in a stable order."""
        schemas = []
        for name in self.enabled:
            description = self.descriptions.get(name, _DEFAULT_DESCRIPTIONS[name])
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": _DEFAULT_PARAMETERS[name],
                    },
                }
            )
        return schemas

    def is_enabled(self, name: str) -> bool:
        return name in self.enabled

    # -------------------------------------------------------------- execute

    def execute(self, call_id: str, tool_name: str, arguments: dict,
                *, clock: Clock) -> ToolResult:
        if tool_name not in self.enabled:
            raise ContractViolation(
                f"tool {tool_name!r} is not enabled for this branch; a disabled "
                "tool must also be absent from the schema, so this call could not "
                "have come from the model"
            )
        if not isinstance(arguments, dict):
            raise ValidationError("tool arguments must be an object")

        handler = _HANDLERS.get(tool_name)
        if handler is None:
            raise ValidationError(f"no handler for tool {tool_name!r}")
        raw = handler(self, arguments, clock)
        visible = self._project(tool_name, raw, call_id)
        return ToolResult(
            call_id=call_id,
            tool_name=tool_name,
            raw=raw,
            visible=visible,
        )

    def _project(self, tool_name: str, raw: Any, call_id: str) -> Any:
        """Apply the fault, if scheduled, then the visibility policy."""
        fault = self.faults.get(call_id) or self.faults.get(tool_name)
        if fault:
            self._applied_faults.add(call_id)
            return _apply_fault(fault, raw)
        return raw

    def applied_faults(self) -> list:
        return sorted(self._applied_faults)

    # -------------------------------------------------------------- handlers

    def _world_time(self, args: dict, clock: Clock) -> dict:
        return {
            "current_time": format_human(clock),
            "time_zone": clock.time_zone,
        }

    def _search(self, args: dict, clock: Clock) -> dict:
        query = args.get("query")
        if not isinstance(query, str) or not query:
            raise ValidationError("search requires a non-empty query string")
        # Result count is an experimental variable, not a model choice.  Letting
        # the model widen `top_k` would let one branch see more documents than
        # another for reasons unrelated to the manipulation, which is exactly the
        # confound the matched-oracle design exists to avoid.  The requested
        # value is recorded but the policy wins.
        requested = args.get("top_k")
        top_k = self.exposure.top_k
        record = self.corpus.retrieve(
            query=query,
            now_instant=clock.instant,
            top_k=top_k,
            mode="probe",
            visibility=self.exposure.mode,
            omit_ids=self.exposure.omit_ids,
            force_include_ids=self.exposure.force_include_ids,
            strip_fields=self.exposure.strip_fields,
            temporal_decay=self.exposure.temporal_decay,
        )
        if isinstance(requested, int) and requested != top_k:
            record.requested_top_k = requested
        self.last_retrieval = record
        documents = []
        for hit in record.hits:
            doc = self.corpus.get(hit["doc_id"])
            entry = {
                "title": doc.source or doc.doc_id,
                "text": doc.text,
                "published": doc.published_at,
            }
            for field in self.exposure.strip_fields:
                entry.pop(field, None)
            documents.append(entry)

        # Annotate each hit with the exact bytes the model will receive.  Audit
        # uses this to prove delivery: "the document was selected" and "the
        # document reached the request" are different facts, and the second one
        # can only be checked against what was actually rendered.
        for hit, entry in zip(record.hits, documents):
            hit["visible_text"] = canonical_json(entry)

        return {"documents": documents}

    def _calendar_events(self, args: dict, clock: Clock) -> dict:
        """Calendar view.  Future entries show that something is scheduled.

        A future event's detail stays out of the visible projection until the
        clock reaches it, which is what makes "scheduled but not yet observable"
        a real state rather than a convention.
        """
        from .clock import parse_instant

        events = []
        now = parse_instant(clock.instant)
        for fact_id, fact in sorted(self.world.facts().items()):
            if not fact_id.startswith("calendar:"):
                continue
            start = fact.get("effective_at")
            future = bool(start) and parse_instant(start) > now
            entry = {
                "title": fact.get("title", fact_id),
                "start": start,
                "status": fact.get("status", "scheduled"),
            }
            if future:
                entry["details_available"] = False
            else:
                entry["details_available"] = True
                if fact.get("location"):
                    entry["location"] = fact["location"]
            events.append(entry)
        return {"events": events}

    def _weather(self, args: dict, clock: Clock) -> dict:
        """Deterministic pseudo-weather from (location, virtual day) only."""
        import hashlib

        location = args.get("location", "local")
        if not isinstance(location, str):
            raise ValidationError("location must be a string")
        day = clock.instant[:10]
        seed = hashlib.sha256(f"{location}|{day}".encode("utf-8")).hexdigest()
        conditions = ["clear", "cloudy", "overcast", "light rain"]
        return {
            "location": location,
            "date": day,
            "condition": conditions[int(seed[:8], 16) % len(conditions)],
            "high_c": 12 + int(seed[8:10], 16) % 18,
            "low_c": 2 + int(seed[10:12], 16) % 10,
        }

    def _official_registry(self, args: dict, clock: Clock) -> dict:
        """Look up an authoritative fact by id."""
        fact_id = args.get("fact_id")
        if not isinstance(fact_id, str) or not fact_id:
            raise ValidationError("official_registry requires a fact_id")
        fact = self.world.fact(fact_id)
        if fact is None:
            return {"found": False}
        return {
            "found": True,
            "value": fact.get("value"),
            "source": fact.get("source"),
            "effective_at": fact.get("effective_at"),
            "revision": fact.get("revision"),
        }


def _apply_fault(fault: str, raw: Any) -> Any:
    """Fault projection.  Only touches the visible side."""
    if fault == "stale_read":
        return {"documents": list(raw.get("documents", []))[1:], "notice": "cached"}
    if fault == "partial_read":
        documents = []
        for doc in raw.get("documents", []):
            documents.append({k: v for k, v in doc.items() if k != "text"})
        return {"documents": documents}
    if fault == "empty_read":
        return {"documents": []}
    raise ValidationError(f"unknown fault {fault!r}")


_DEFAULT_DESCRIPTIONS = {
    "world_time": "Get the current date and time.",
    "search": "Search your notes and documents for relevant information.",
    "calendar_events": "List calendar entries and their scheduled times.",
    "weather": "Get the weather forecast for a location.",
    "official_registry": "Look up an official record by its identifier.",
}

_DEFAULT_PARAMETERS = {
    "world_time": {"type": "object", "properties": {}},
    "search": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "How many results to return."},
        },
        "required": ["query"],
    },
    "calendar_events": {"type": "object", "properties": {}},
    "weather": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
    },
    "official_registry": {
        "type": "object",
        "properties": {"fact_id": {"type": "string"}},
        "required": ["fact_id"],
    },
}

_HANDLERS: dict[str, Callable] = {
    "world_time": lambda belt, args, clock: belt._world_time(args, clock),
    "search": lambda belt, args, clock: belt._search(args, clock),
    "calendar_events": lambda belt, args, clock: belt._calendar_events(args, clock),
    "weather": lambda belt, args, clock: belt._weather(args, clock),
    "official_registry": lambda belt, args, clock: belt._official_registry(args, clock),
}
