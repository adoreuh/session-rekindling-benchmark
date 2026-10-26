"""Error vocabulary.

Every failure here is *fail-closed*: the harness would rather stop than
continue with a state it cannot vouch for.  ``ContractViolation`` is the
important one — it marks a break in experiment semantics, which is never a
model-behaviour observation.
"""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for all harness errors."""


class ValidationError(HarnessError, ValueError):
    """Malformed input: bad JSON, bad schema, bad field type."""


class ContractViolation(HarnessError):
    """A break in experiment semantics.

    Raised when the harness detects that a guarantee in ``CONTRACT.md`` has
    been violated: a domain gate crossed, a prompt assembled at runtime, an
    unregistered world event, a fork of a non-stable boundary, and so on.

    These must never be downgraded to a benign tool error handed back to the
    model.  They invalidate the run.
    """


class GateRejection(ContractViolation):
    """A pre-dispatch check rejected a request that was about to be sent.

    Carries the verdict rather than only a message, because the *scope* of the
    rejection decides how far the damage reaches: a request-level rejection
    invalidates one branch, while a batch-level one (a contaminated parent, an
    unauditable serialization path) invalidates every branch derived from the
    same state.  Collapsing the two into one exception would force the runner to
    guess, and guessing wrong is how a shared defect gets reported as a single
    branch's bad luck.
    """

    def __init__(self, message: str, *, verdict, scope: str = "request"):
        super().__init__(message)
        self.verdict = verdict
        self.scope = scope


class IntegrityError(HarnessError):
    """The persisted log failed verification (tamper, truncation, mismatch)."""


class StaleWriterError(HarnessError):
    """A detached writer tried to append to a log that moved on."""


class ProviderError(HarnessError):
    """The provider call failed in a way that is not a model behaviour.

    Carries ``attempts`` so the run record can distinguish "the model refused"
    from "the endpoint broke".
    """

    def __init__(self, message: str, *, attempts: int = 0, retryable: bool = False):
        super().__init__(message)
        self.attempts = attempts
        self.retryable = retryable


class AuditFailure(HarnessError):
    """Offline audit found a violation.  Carries structured findings."""

    def __init__(self, findings):
        self.findings = list(findings)
        rendered = "; ".join(f"{f['check']}: {f['detail']}" for f in self.findings)
        super().__init__(f"audit failed with {len(self.findings)} finding(s): {rendered}")
