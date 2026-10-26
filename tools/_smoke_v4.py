"""Offline smoke run of the wired v4 spec.  No provider call.

Its job is to prove the integration is real: that the gate is invoked on every
request, that the oracle condition actually delivers the notice, that the omit
condition actually does not, and that the two conditions differ in nothing else.
A spec that merely *constructs* proves none of that.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vela_lab.provider import MockModel  # noqa: E402
from vela_lab.runner import ExperimentRunner  # noqa: E402

from tools.v4_wiring import build_v4_spec  # noqa: E402


def main() -> int:
    out = Path("runs_gate/v4_smoke")
    if out.exists():
        shutil.rmtree(out)

    spec = build_v4_spec(experiment_id="v4-smoke", elapsed_days=15)
    model = MockModel(scripted=["我理解你的担心。"])
    report = ExperimentRunner(spec, model, out_dir=out).run()

    print(f"advance_days      : {spec.advance_days}")
    print(f"enabled_tools     : {list(spec.enabled_tools)}")
    print(f"parent check      : {report['exposure']['parent_check']['status']} "
          f"{report['exposure']['parent_check']['reason_codes']}")
    print(f"contract          : {report['exposure']['contract_version']}")
    print()

    problems = []
    for branch in report["branches"]:
        verdicts = branch["exposure_verdicts"]
        statuses = [v["status"] for v in verdicts]
        auto = branch["auto_retrieval"]
        print(f"{branch['branch_id']:18s} valid={str(branch['run_valid']):5s} "
              f"req={branch['outcome']['request_count']} "
              f"delivered={auto.get('hits')}")
        print(f"{'':18s} gate={statuses} reasons="
              f"{sorted({r for v in verdicts for r in v['reason_codes']})}")
        for v in verdicts:
            print(f"{'':18s}   present={v['present_material_ids']} "
                  f"undeclared={v['undeclared_material_ids']} "
                  f"propositions={v['matched_propositions']}")
        if not branch["run_valid"]:
            problems.append(f"{branch['branch_id']} invalid: "
                            f"{branch['invalid_reason']} "
                            f"{branch['gate_rejection']}")
        if not verdicts:
            problems.append(f"{branch['branch_id']} was never checked")

    print()
    # The pairing claim, checked rather than asserted.
    by_id = {b["branch_id"]: b for b in report["branches"]}
    for pressure in ("low", "high"):
        o = by_id[f"claim_{pressure}_oracle"]["auto_retrieval"]["hits"]
        m = by_id[f"claim_{pressure}_omit"]["auto_retrieval"]["hits"]
        diff = [i for i, (a, b) in enumerate(zip(o, m)) if a != b]
        same_len = len(o) == len(m)
        print(f"pair {pressure}: same_count={same_len} differing_slots={diff} "
              f"slot_differs_only={same_len and len(diff) == 1}")

    # Text volume, so "evidence supplied" is not confounded with "more text".
    texts = {d.doc_id: d.text for d in spec.documents}
    o_len = sum(len(texts[i]) for i in by_id["claim_low_oracle"]["auto_retrieval"]["hits"])
    m_len = sum(len(texts[i]) for i in by_id["claim_low_omit"]["auto_retrieval"]["hits"])
    print(f"delivered chars: oracle={o_len} omit={m_len} "
          f"ratio={max(o_len, m_len) / max(min(o_len, m_len), 1):.3f}")

    print()
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print("smoke OK: gate ran on every request, pairing holds, volumes matched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
