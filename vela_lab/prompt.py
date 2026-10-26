"""Registered, frozen prompt bundles.

This module exists to make a second specific bug impossible.

The project's bridging experiment produced a result that was entirely an
artifact of one sentence.  With ``请以事实为准`` ("defer to the facts") in the
system prompt, contradiction detection measured 85.7% and adoption 0%.  With the
sentence removed, detection was 0% and adoption 85.7%.  One well-meaning clause
decided the direction of the finding.  The clause was added by hand, at runtime,
because nothing stopped it.

So: **the run entry point does not accept a system string.**  It accepts a
``prompt_id`` naming a bundle that was registered and hashed before the run
started.  A bundle carries the system message, every tool description, the
memory-injection wrapper, and the error templates — i.e. everything that trips
the model.  A bundle is immutable once registered; changing one character and
re-registering yields a different digest, which the manifest records and the
audit compares.

Honest limit: this proves the prompt was not edited between registration and
dispatch.  It does *not* prove that natural language is behaviourally inert.  If
a conclusion turns out to be sensitive to wording, that has to be measured and
reported as a scope limitation — see ``sensitivity_probe`` below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .errors import ContractViolation, ValidationError
from .jsonutil import digest

BUNDLE_FIELDS = frozenset(
    {
        "prompt_id",
        "system",
        "tool_descriptions",
        "memory_wrapper",
        "error_templates",
        "notes",
    }
)


@dataclass(frozen=True)
class PromptBundle:
    """A named, digestible set of everything that reaches the model's prompt.

    ``digest()`` covers the whole bundle, so any edit — including a whitespace
    change in a tool description — produces a different identity.
    """

    prompt_id: str
    system: str
    tool_descriptions: Mapping[str, str] = field(default_factory=dict)
    memory_wrapper: str = "Relevant notes from earlier conversations:\n{entries}"
    error_templates: Mapping[str, str] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self):
        if not isinstance(self.prompt_id, str) or not self.prompt_id:
            raise ValidationError("prompt_id must be a non-empty string")
        if not isinstance(self.system, str) or not self.system.strip():
            raise ValidationError("system prompt must be non-empty")
        for name, text in self.tool_descriptions.items():
            if not isinstance(text, str) or not text.strip():
                raise ValidationError(f"tool description for {name!r} must be non-empty")
        for name, text in self.error_templates.items():
            if not isinstance(text, str):
                raise ValidationError(f"error template {name!r} must be a string")

    def identity(self) -> dict:
        """The hashed projection.  Order-insensitive via sorted mappings."""
        return {
            "prompt_id": self.prompt_id,
            "system": self.system,
            "tool_descriptions": dict(sorted(self.tool_descriptions.items())),
            "memory_wrapper": self.memory_wrapper,
            "error_templates": dict(sorted(self.error_templates.items())),
        }

    def digest(self) -> str:
        return digest(self.identity())

    def to_json(self) -> dict:
        return {**self.identity(), "notes": self.notes}

    @staticmethod
    def from_json(payload: dict) -> "PromptBundle":
        unknown = set(payload) - BUNDLE_FIELDS
        if unknown:
            raise ValidationError(f"prompt bundle: unknown keys {sorted(unknown)}")
        missing = {"prompt_id", "system"} - set(payload)
        if missing:
            raise ValidationError(f"prompt bundle: missing {sorted(missing)}")
        return PromptBundle(
            prompt_id=payload["prompt_id"],
            system=payload["system"],
            tool_descriptions=dict(payload.get("tool_descriptions", {})),
            memory_wrapper=payload.get(
                "memory_wrapper", "Relevant notes from earlier conversations:\n{entries}"
            ),
            error_templates=dict(payload.get("error_templates", {})),
            notes=payload.get("notes", ""),
        )


class PromptRegistry:
    """Holds registered bundles.  Registration is one-way.

    Re-registering the same id with different content raises rather than
    silently replacing: a run must not be able to swap its prompt after the
    point where anybody could have reviewed it.
    """

    def __init__(self):
        self._bundles: dict[str, PromptBundle] = {}
        self._sealed = False

    def register(self, bundle: PromptBundle) -> str:
        if self._sealed:
            raise ContractViolation(
                "prompt registry is sealed; registration closed before the run "
                "started and no bundle may be added or replaced afterwards"
            )
        existing = self._bundles.get(bundle.prompt_id)
        if existing is not None:
            if existing.digest() != bundle.digest():
                raise ContractViolation(
                    f"prompt {bundle.prompt_id!r} already registered with a "
                    f"different digest ({existing.digest()[:12]} vs "
                    f"{bundle.digest()[:12]}); prompts are immutable — use a new id"
                )
            return existing.digest()
        self._bundles[bundle.prompt_id] = bundle
        return bundle.digest()

    def seal(self) -> None:
        """Close registration.  Called once the run's prompts are all in."""
        self._sealed = True

    def get(self, prompt_id: str) -> PromptBundle:
        bundle = self._bundles.get(prompt_id)
        if bundle is None:
            raise ContractViolation(
                f"no registered prompt bundle {prompt_id!r}; prompts must be "
                "registered before the run, never assembled at runtime"
            )
        return bundle

    def digests(self) -> dict:
        return {pid: b.digest() for pid, b in sorted(self._bundles.items())}

    def ids(self) -> list:
        return sorted(self._bundles)


def neutral_system_prompt() -> str:
    """A deliberately plain baseline prompt.

    No stance on facts, affect, accuracy, or the user's reliability.  The
    bridging experiment showed that any such instruction is an intervention, so
    the baseline contains none.
    """
    return (
        "You are a helpful assistant in an ongoing conversation with a user. "
        "Use the tools available to you when they are useful. "
        "Reply in the user's language."
    )


def sensitivity_report(bundles: Mapping[str, PromptBundle]) -> dict:
    """Report which prompt clauses differ across bundles.

    Feeds the "is the conclusion wording-sensitive?" check.  Identifies clauses
    present in some bundles and absent in others, so a reviewer can see exactly
    what varied rather than trusting a summary.
    """
    systems = {pid: b.system for pid, b in bundles.items()}
    if len(set(systems.values())) <= 1:
        return {"varies": False, "bundles": sorted(systems)}
    sentences: dict[str, set] = {}
    for pid, text in systems.items():
        sentences[pid] = {
            part.strip() for part in text.replace("\n", " ").split(".") if part.strip()
        }
    common = set.intersection(*sentences.values()) if sentences else set()
    differing = {
        pid: sorted(parts - common) for pid, parts in sentences.items() if parts - common
    }
    return {
        "varies": True,
        "bundles": sorted(systems),
        "common": sorted(common),
        "differing": differing,
    }
