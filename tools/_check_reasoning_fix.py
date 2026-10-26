"""Verify the reasoning-content fix against the route that rejected a continuation.

The failure this reproduces is provider-shaped, not behavioural: DeepSeek's
thinking mode returns HTTP 400 on a follow-up tool turn unless the assistant
message carries back the ``reasoning_content`` it sent.  So the test forces a
tool round and checks the second request is accepted.

Before the fix this raised, and the run was recorded as a failure that had nothing
to do with the model's answer.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela_lab.provider import OpenAICompatibleModel, MockModel  # noqa: E402
from vela_lab.runner import ExperimentRunner  # noqa: E402

from tools.scenario_factory import FILLER_ID, PROFILES, build_variant  # noqa: E402
from tools.variant_wiring import build_variant_spec  # noqa: E402


def forced_tool_call_response(*, model: str, base_url: str, key_env: str):
    """A provider whose first answer requests a tool, to force a continuation."""

    class Forcing(OpenAICompatibleModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.calls = 0

        def complete(self, payload):
            self.calls += 1
            if self.calls == 1:
                return MockModel(
                    scripted=[{
                        "content": "",
                        "finish_reason": "tool_calls",
                        "tool_calls": [{
                            "id": "c1", "type": "function",
                            "function": {"name": "world_time", "arguments": "{}"},
                        }],
                    }]
                ).complete(payload)
            return super().complete(payload)

    return Forcing(base_url=base_url, model=model, api_key_env=key_env,
                   timeout=300, max_transport_retries=1)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash")
    parser.add_argument("--base-url", default="__REDACTED_BASE_URL__")
    parser.add_argument("--api-key-env", default="VELA_API_KEY")
    args = parser.parse_args()

    variant = build_variant(scenario_id="t", profile=PROFILES[0],
                            filler_id=FILLER_ID)
    spec = build_variant_spec(variant=variant, experiment_id="reasoning-fix",
                              elapsed_days=15)
    # One branch is enough: the point is whether a continuation is accepted.
    spec.branches = tuple(b for b in spec.branches
                          if b.branch_id == "claim_low_oracle")

    out = Path("runs_gate/reasoning_fix")
    if out.exists():
        shutil.rmtree(out)
    model = forced_tool_call_response(
        model=args.model, base_url=args.base_url, key_env=args.api_key_env
    )
    report = ExperimentRunner(spec, model, out_dir=out).run()
    branch = report["branches"][0]
    requests = branch["outcome"]["request_count"]
    failure = (branch["outcome"].get("failure") or {}).get("reason", "")

    print(f"model      : {args.model}")
    print(f"requests   : {requests} (2 means the continuation was accepted)")
    print(f"tool_calls : {branch['outcome']['tool_calls']}")
    print(f"failure    : {failure or 'none'}")
    print(f"answer     : {branch['outcome']['assistant_text'][:120]!r}")

    # Also confirm the reasoning text was actually captured and replayed.
    log = out / "branches" / "claim_low_oracle" / "events.jsonl"
    captured = 0
    replayed = 0
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event["type"] == "observation.compiled":
            if event["data"].get("reasoning_content"):
                captured += 1
        if event["type"] == "model.request":
            for message in event["data"]["payload"].get("messages", []):
                if message.get("reasoning_content"):
                    replayed += 1
    print(f"reasoning captured in log : {captured}")
    print(f"reasoning replayed in request: {replayed}")

    ok = requests >= 2 and not failure
    print()
    print("PASS: the tool continuation was accepted" if ok
          else "FAIL: the continuation still fails")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
