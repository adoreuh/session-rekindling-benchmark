"""Frozen run policy, with a consistency gate for resuming.

The problem this solves is narrow but real, and it has already bitten this
project twice.

Concretely, if a run is continued under a newer *default* configuration:

* the old run retrieved with ``top_k=2`` and the continuation retrieves with
  ``top_k=4``, so a key document that ranked third moves from unreachable to
  reachable and both the accessibility and attribution labels change;
* the old run injected notes permanently and the continuation retires them at
  the turn boundary, so two conditions differ in the strength of the
  accessibility manipulation while being counted as the same experiment;
* a wrapper or prompt default gains a clause like "defer to the facts" — and the
  project's own bridging experiment showed that a single such clause flipped
  detection from 0% to 85.7%.

None of those need a general policy-recovery engine to prevent. What they need
is that the policy be **resolved once, frozen, referenced by every call, and
checked for agreement before a run continues**. Continuing under a different
policy is refused; the caller starts a new run instead.

So this module deliberately does *not* implement history-aware policy
restoration. It implements:
``resolve once`` → ``freeze`` → ``verify before resume`` → ``refuse on mismatch``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .errors import ContractViolation
from .jsonutil import digest, json_copy

# Fields that must be identical for a run to be continued.  Kept explicit rather
# than "hash everything", so that adding an unrelated debug field does not
# invalidate every historical run.
CONSISTENCY_FIELDS = (
    "prompt_digest",
    "tool_schemas_digest",
    "enabled_tools",
    "phase_top_k",
    "note_lifetime",
    "memory_wrapper_digest",
    "tool_budget",
    "request_params",
    "model_identity",
    "corpus_digest",
    "schedule_digest",
)


@dataclass(frozen=True)
class RunPolicy:
    """The resolved, immutable policy for one run.

    ``phase_top_k`` is a mapping from phase name to result width.  Keeping it per
    phase lets a prefix legitimately use a different width from a probe while
    still being frozen, recorded, and checked — which matters because the two
    phases are jointly experienced by every branch, so a difference between them
    is not a branch-level confound, whereas an *unrecorded* difference is.
    """

    prompt_digest: str
    tool_schemas_digest: str
    enabled_tools: tuple
    phase_top_k: Mapping[str, int]
    note_lifetime: str
    memory_wrapper_digest: str
    tool_budget: int
    request_params: Mapping[str, Any]
    model_identity: Mapping[str, Any]
    corpus_digest: str = ""
    schedule_digest: str = ""
    protocol_version: str = ""

    def identity(self) -> dict:
        return {
            "prompt_digest": self.prompt_digest,
            "tool_schemas_digest": self.tool_schemas_digest,
            "enabled_tools": list(self.enabled_tools),
            "phase_top_k": dict(sorted(self.phase_top_k.items())),
            "note_lifetime": self.note_lifetime,
            "memory_wrapper_digest": self.memory_wrapper_digest,
            "tool_budget": self.tool_budget,
            "request_params": dict(sorted(self.request_params.items())),
            "model_identity": dict(sorted(self.model_identity.items())),
            "corpus_digest": self.corpus_digest,
            "schedule_digest": self.schedule_digest,
            "protocol_version": self.protocol_version,
        }

    def digest(self) -> str:
        return digest(self.identity())

    def top_k_for(self, phase: str) -> int:
        if phase not in self.phase_top_k:
            raise ContractViolation(
                f"no top_k registered for phase {phase!r}; the policy is frozen "
                f"for {sorted(self.phase_top_k)}"
            )
        return int(self.phase_top_k[phase])

    def to_json(self) -> dict:
        return self.identity()

    @staticmethod
    def from_json(payload: dict) -> "RunPolicy":
        missing = [f for f in CONSISTENCY_FIELDS if f not in payload]
        if missing:
            raise ContractViolation(
                f"run policy record is missing {missing}; a policy that cannot be "
                "read back cannot be checked for agreement"
            )
        return RunPolicy(
            prompt_digest=payload["prompt_digest"],
            tool_schemas_digest=payload["tool_schemas_digest"],
            enabled_tools=tuple(payload["enabled_tools"]),
            phase_top_k=dict(payload["phase_top_k"]),
            note_lifetime=payload["note_lifetime"],
            memory_wrapper_digest=payload["memory_wrapper_digest"],
            tool_budget=payload["tool_budget"],
            request_params=json_copy(payload["request_params"], "request params"),
            model_identity=json_copy(payload["model_identity"], "model identity"),
            corpus_digest=payload.get("corpus_digest", ""),
            schedule_digest=payload.get("schedule_digest", ""),
            protocol_version=payload.get("protocol_version", ""),
        )


def compare_policies(recorded: RunPolicy, current: RunPolicy) -> list:
    """Report which consistency fields differ.  Empty means it is safe to resume.

    Compares field by field rather than digest to digest, so the message names
    the offending field instead of only saying "the policy changed".
    """
    findings = []
    left, right = recorded.identity(), current.identity()
    for name in CONSISTENCY_FIELDS:
        if left.get(name) != right.get(name):
            findings.append(
                {
                    "field": name,
                    "recorded": left.get(name),
                    "current": right.get(name),
                }
            )
    return findings


def assert_resumable(recorded: RunPolicy, current: RunPolicy) -> None:
    """Fail closed when a run would continue under a different policy."""
    findings = compare_policies(recorded, current)
    if findings:
        detail = "; ".join(
            f"{f['field']}: {f['recorded']!r} -> {f['current']!r}" for f in findings
        )
        raise ContractViolation(
            "refusing to continue this run: the resolved policy differs from the "
            f"one it was started with ({detail}). Continuing would mix two "
            "conditions under one name. Start a new run instead."
        )
