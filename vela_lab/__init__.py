"""VELA Lab — a controlled world-time / state-validity experiment harness.

Design authority: ``CONTRACT.md`` in the repository root.  The contract is the
experiment-semantics document; code that contradicts it is a bug.

Layout (responsibilities, not layers):

* ``jsonutil``   strict JSON + canonical encoding + hashing
* ``clock``      the single virtual clock
* ``log``        append-only hash-chained event log
* ``reducer``    event vocabulary + pure folding into ``DomainState``
* ``domains``    the four state domains and the access gates between them
* ``session``    checkpoint / fork over the log
* ``prompt``     registered, frozen prompt bundles
* ``compiler``   the single request outlet, with per-fragment provenance
* ``policy``     the frozen run policy and the resume consistency gate
* ``schedule``   registered world-event table and deterministic scheduling
* ``evidence``   controlled evidence corpus and the exposure chain
* ``tools``      read-only tool registry with raw/visible separation
* ``provider``   model adapters that record failures instead of hiding them
* ``loop``       the minimal agent turn loop
* ``audit``      fail-closed offline verification
* ``score``      L1-L4 dependent-variable extraction
"""

__all__ = [
    "audit",
    "clock",
    "compiler",
    "domains",
    "evidence",
    "jsonutil",
    "log",
    "loop",
    "policy",
    "prompt",
    "provider",
    "reducer",
    "schedule",
    "score",
    "session",
    "tools",
]

PROTOCOL_VERSION = "vela-lab-protocol-v0.4"
