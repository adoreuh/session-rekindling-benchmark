"""Session: the kernel that folds events, plus checkpoint and fork.

Contract §5: every experimental condition forks from the *same* stable parent.
This module is where that promise is kept or broken.

Checkpoint.  A checkpoint is only legal at a stable boundary — no open turn, no
half-finished tool call.  The payload records the folded state, the observation
list, the prompt binding, and the event prefix (count and chain digest).  The
``checkpoint.created`` event is written **after** the payload is durably in
place, so a crash cannot leave a marker pointing at a snapshot that does not
exist.  A marker without its payload would let a later reader believe in a
boundary that was never captured.

Fork.  A fork copies the parent's event prefix byte-for-byte into a new log,
then re-derives state from it independently.  Two consequences matter:

* the child is self-contained — it can be moved, archived, or audited after the
  parent is deleted;
* the child does not inherit the parent's *trust*, only its bytes.  State is
  recomputed and compared, so a corrupted parent fails the fork instead of
  propagating.

The cost is storage.  That cost is accepted here on purpose: a branch that
cannot be audited alone is not evidence.  A content-addressed prefix store is
the right optimisation later, and it must keep the prefix objects alive for as
long as any branch references them.

Branch writes are branch-local.  In a diagnostic comparison, a probe branch must
not write back into the parent — otherwise an adopted false claim becomes a
permanent memory mutation, and the run stops measuring compliance and starts
measuring contamination, which is a different (and already-published) question.
"""

from __future__ import annotations

from pathlib import Path

from .clock import Clock
from .domains import ObservationDomain, WorldDomain
from .errors import ContractViolation, IntegrityError, ValidationError
from .jsonutil import digest, json_copy
from .log import EventLog, verify_chain
from .reducer import KernelState, initial_state, reduce_all, reduce_event

CHECKPOINT_DIR = "checkpoints"
SNAPSHOT_NAME = "snapshot.json"


class Session:
    """A single experimental trajectory.

    Wraps an ``EventLog`` and keeps a folded ``KernelState`` in step with it.
    Appends validate against a *proposed* state first, so a rejected event never
    mutates the live view.
    """

    def __init__(
        self,
        root,
        *,
        clock: Clock,
        observations: ObservationDomain | None = None,
        world: WorldDomain | None = None,
        config: dict | None = None,
    ):
        self.root = Path(root)
        self.log = EventLog(self.root)
        self.observations = observations or ObservationDomain()
        self.world = world or WorldDomain()
        self.config = json_copy(config or {}, "config")
        self._request_counter = 0

        if self.log.head().seq == 0:
            self.state = initial_state(clock)
            self.append(
                "session.started",
                {"clock": clock.to_json(), "config": self.config},
            )
        else:
            self.state = self._fold(self.log.read())
            self._request_counter = len(self.state.requests)
        self.sync_world()

    def sync_world(self) -> None:
        """Bring the ``WorldDomain`` view in step with the folded state.

        Facts have exactly one authority — the event log.  The domain object is
        a read-only view rebuilt from it.
        """
        self.world.restore_from_facts(self.state.facts)

    # ------------------------------------------------------------- properties

    @property
    def clock(self) -> Clock:
        if self.state.clock is None:
            raise ContractViolation("session has no clock")
        return self.state.clock

    def head_seq(self) -> int:
        return self.log.head().seq

    def next_request_index(self) -> int:
        self._request_counter += 1
        return self._request_counter

    def events(self) -> list:
        return self.log.read()

    # ---------------------------------------------------------------- append

    def append(self, event_type: str, data: dict) -> dict:
        """Validate against a proposed state, persist, then commit."""
        event = {
            "seq": self.log.head().seq + 1,
            "type": event_type,
            "data": json_copy(data, "event data"),
        }
        proposed = reduce_event(self.state, event)
        stored = self.log.append(
            event_type, event["data"], virtual_time=self._virtual_time()
        )
        self.state = proposed
        if event_type in ("world.fact_set", "world.fact_superseded"):
            self.sync_world()
        return stored

    def _virtual_time(self) -> str:
        return self.state.clock.instant if self.state.clock else ""

    def _fold(self, events: list) -> KernelState:
        if not events:
            raise IntegrityError("cannot fold an empty log")
        state = initial_state(Clock.of("1970-01-01T00:00:00Z", "UTC"))
        return reduce_all(events, state)

    # ------------------------------------------------------------- time

    def _at_boundary(self) -> bool:
        """Both the event stream and the observation turn must be closed.

        The event stream knows about ``turn.started``/``turn.completed``; the
        observation domain knows whether a user turn has been opened and not yet
        closed.  A boundary needs both, or a user turn would span a time
        advance, a checkpoint, or a session change.
        """
        return self.state.is_stable() and not self.observations.turn_open

    def advance(self, *, days: int = 0, hours: int = 0, minutes: int = 0) -> Clock:
        """Move the virtual clock, at a stable boundary only.

        ``schedule`` handling is not implicit: the caller decides which
        registered events fire, which keeps world changes explicit and logged.
        """
        if not self._at_boundary():
            raise ContractViolation(
                "the clock may only advance at a stable boundary; advancing "
                "mid-turn would let a single turn span two conditions"
            )
        target = self.clock.advance(days=days, hours=hours, minutes=minutes)
        self.append(
            "clock.advance",
            {"to": target.instant, "time_zone": target.time_zone},
        )
        return self.clock

    # ---------------------------------------------------------- checkpoint

    def checkpoint(self, name: str) -> dict:
        if not self._at_boundary():
            raise ContractViolation(
                f"cannot checkpoint {name!r}: the session is not at a stable "
                "boundary (open turn or unresolved tool call)"
            )
        events = self.log.read()
        snapshot = {
            "name": name,
            "state": self.state.to_json(),
            "observations": self.observations.snapshot(),
            "window": self.observations.window,
            "expired_observations": [
                f.to_json() for f in self.observations.expired_observations()
            ],
            "world": self.world.snapshot(),
            "config": self.config,
            "prefix_count": len(events),
            "prefix_digest": digest([e["hash"] for e in events]),
            "prefix_head": events[-1]["hash"] if events else "",
        }
        snapshot["checksum"] = digest(
            {k: v for k, v in snapshot.items() if k != "checksum"}
        )

        directory = self.root / CHECKPOINT_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.json"
        if path.exists():
            raise ContractViolation(f"checkpoint {name!r} already exists")

        staged = path.with_suffix(".json.staged")
        import json

        with staged.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(snapshot, ensure_ascii=False, sort_keys=True))
            handle.flush()
            import os

            os.fsync(handle.fileno())
        staged.replace(path)

        # Only now is the marker written.  A marker without a payload would be
        # a claim about a boundary that was never captured.
        self.append(
            "checkpoint.created",
            {
                "name": name,
                "prefix_count": snapshot["prefix_count"],
                "prefix_digest": snapshot["prefix_digest"],
                "checksum": snapshot["checksum"],
            },
        )
        return snapshot

    def load_checkpoint(self, source_root, name: str) -> dict:
        """Read and verify a checkpoint from another session root."""
        path = Path(source_root) / CHECKPOINT_DIR / f"{name}.json"
        if not path.exists():
            raise ValidationError(f"no checkpoint {name!r} under {source_root}")
        import json

        snapshot = json.loads(path.read_text(encoding="utf-8"))
        expected = digest({k: v for k, v in snapshot.items() if k != "checksum"})
        if expected != snapshot.get("checksum"):
            raise IntegrityError(
                f"checkpoint {name!r}: checksum mismatch; the snapshot was edited"
            )
        return snapshot

    def fork(self, source_root, name: str, *, target_root=None, world=None) -> "Session":
        """Create an independent child session from a parent checkpoint.

        The parent's event prefix is copied byte-for-byte and then re-folded and
        compared against the checkpoint's recorded state.  A disagreement means
        the parent was edited, and the fork fails rather than inheriting it.
        """
        snapshot = self.load_checkpoint(source_root, name)
        target_root = Path(target_root or self.root)

        source_log = EventLog(Path(source_root))
        source_events = source_log.read()
        prefix_count = snapshot["prefix_count"]
        if prefix_count > len(source_events):
            raise IntegrityError(
                f"checkpoint {name!r} claims {prefix_count} events but the parent "
                f"log has only {len(source_events)}"
            )
        prefix = source_events[:prefix_count]
        verify_chain(prefix)
        if digest([e["hash"] for e in prefix]) != snapshot["prefix_digest"]:
            raise IntegrityError(
                f"checkpoint {name!r}: prefix digest does not match the parent log"
            )

        # Copy, then re-derive independently.
        if target_root.exists() and (target_root / "events.jsonl").exists():
            raise ContractViolation(f"{target_root} already holds a log")
        target_root.mkdir(parents=True, exist_ok=True)
        temp = EventLog(target_root)
        import json

        payload = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n"
            for record in prefix
        )
        with temp.path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            import os

            os.fsync(handle.fileno())
        verify_chain(temp.read())

        child_world = world or WorldDomain()
        child_world.restore(snapshot["world"])
        child = Session.__new__(Session)
        child.root = target_root
        child.log = EventLog(target_root)
        child.observations = ObservationDomain(
            window=int(snapshot.get("window", 0))
        )
        child.observations.restore(snapshot["observations"])
        for record in snapshot.get("expired_observations", []):
            # Expired fragments are restored directly: they were already
            # validated when first minted, and re-minting them as active would
            # wrongly make previously-exposed evidence available again.
            from .domains import Provenance as _Prov, _mint as _mk

            child.observations._expired.append(
                _mk(
                    record["kind"],
                    record["text"],
                    _Prov(
                        source=record["provenance"]["source"],
                        ref=record["provenance"]["ref"],
                        digest=record["provenance"].get("digest", ""),
                    ),
                    meta=record.get("meta", {}),
                    window=int(record.get("window", 0)),
                    frozen=bool(record.get("frozen", False)),
                )
            )
        child.world = child_world
        child.config = json_copy(snapshot["config"], "config")
        # The child inherits the prefix's request count.  Restarting at zero
        # would make the prefix's request and the branch's request share an
        # index, and "which request was this?" would become unanswerable.
        child._request_counter = 0

        derived = reduce_all(child.log.read(), initial_state(
            Clock.of("1970-01-01T00:00:00Z", "UTC")
        ))
        recorded = KernelState.from_json(snapshot["state"])
        if derived.digest() != recorded.digest():
            raise IntegrityError(
                f"checkpoint {name!r}: re-derived state differs from the recorded "
                "snapshot; the parent log and its checkpoint disagree"
            )
        child.state = derived
        child.sync_world()
        # Continue the request sequence from the inherited prefix.
        child._request_counter = len(derived.requests)

        child.append(
            "branch.forked",
            {
                "checkpoint": name,
                "parent_prefix_count": prefix_count,
                "parent_prefix_digest": snapshot["prefix_digest"],
                "pre_state_digest": recorded.digest(),
            },
        )
        return child

    # ------------------------------------------------------------- replay

    def replay(self) -> KernelState:
        """Rebuild state from the log with no provider and no side effects."""
        events = self.log.read()
        return reduce_all(
            events, initial_state(Clock.of("1970-01-01T00:00:00Z", "UTC"))
        )

    def verify(self) -> list:
        """Compare the live state against a fresh fold.  Returns findings."""
        findings = []
        replayed = self.replay()
        if replayed.digest() != self.state.digest():
            findings.append(
                {
                    "check": "state_matches_replay",
                    "detail": "in-memory state differs from the log-derived state",
                }
            )
        return findings

    # ------------------------------------------------------------- windows

    def register_stable_material(self, material_id: str, text: str) -> None:
        """Declare pre-registered background that outlives every window.

        Must be registered before any window switch.  Freezing material means it
        can never be the object of an accessibility manipulation, so the call is
        recorded as its own event rather than being folded into session start:
        a reader has to be able to see exactly which facts were placed
        permanently in context.
        """
        fragment = self.observations.stable_material(text, material_id=material_id)
        self.append(
            "stable_material.registered",
            {
                "material_id": material_id,
                "text": text,
                "content_digest": digest(text),
                "provenance": fragment.provenance.to_json(),
            },
        )

    def close_window(self) -> None:
        """Close the current session window, retiring its turn-scoped notes.

        A window may only close at a stable boundary.  Closing mid-turn would
        mean a user turn spanned two sessions, which is neither a real
        interaction nor analysable.
        """
        if not self._at_boundary():
            raise ContractViolation(
                "a session window may only close at a stable boundary; closing "
                "mid-turn would let one user turn span two sessions"
            )
        self.observations.end_turn()
        self.append("session.closed", {"window": self.observations.window})

    def open_window(self) -> dict:
        """Open the next session window and rebase the visible context.

        Order matters and is enforced by the reducer: ``session.closed``, then
        ``context.rebased``, then ``session.opened``.  The rebase is recorded
        *before* the window officially opens so that the manifest describing what
        was dropped belongs to the window it applies to.
        """
        self.observations.begin_window()
        manifest = self.observations.rebase()
        self.append("context.rebased", manifest)
        self.append("session.opened", {"window": self.observations.window})
        return manifest

    def switch_window(self) -> dict:
        """Close the current window and open the next one, rebasing context."""
        self.close_window()
        return self.open_window()

    def add_user_turn(self, text: str) -> None:
        """Deliver a user message: the only way user text enters history.

        Opening a turn here means a previous turn's memory notes have already
        been retired — evidence exposed last turn is no longer active, which is
        what makes decay out of reach expressible at all.
        """
        self.observations.begin_turn()
        seq = self.head_seq() + 1
        fragment = self.observations.user(text, event_seq=seq)
        self.append(
            "user.message",
            {
                "kind": fragment.kind,
                "text": text,
                "provenance": fragment.provenance.to_json(),
                "window": fragment.window,
            },
        )

    def add_time_context(self) -> None:
        """Deliver the current time as a visible observation."""
        from .reducer import render_time_context

        text = render_time_context(self.clock)
        seq = self.head_seq() + 1
        fragment = self.observations.time_context(
            text, event_seq=seq, digest_=digest(text)
        )
        self.append(
            "observation.compiled",
            {
                "kind": fragment.kind,
                "text": text,
                "provenance": fragment.provenance.to_json(),
                "window": fragment.window,
            },
        )
