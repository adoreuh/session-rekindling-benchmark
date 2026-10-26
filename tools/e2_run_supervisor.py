"""Run the E2 grid route-by-route, retrying failed cells until coverage is full."""
import os, subprocess, sys, time
from pathlib import Path

HARN = Path(__file__).resolve().parents[1]
os.chdir(HARN)
env = dict(os.environ)
env.setdefault("VELA_API_KEY", os.environ.get("VELA_API_KEY", ""))
env["TEMP"] = str(HARN / "_tmp")

PROFILES = ("visa_interview_i1 driving_test_i1 medical_exam_i1 thesis_defense_i1 "
            "house_viewing_i1 language_test_i1 equipment_check_i1 immigration_intake_i1 "
            "broadband_install_i1 car_maintenance_i1 employment_checkup_i1 furniture_measure_i1 "
            "homestay_checkin_i1 kindergarten_interview_i1 ielts_registration_i1 "
            "lease_signing_i1 loan_interview_i1 specialist_clinic_i1 tax_interview_i1").split()
ROUTES = ("hy3", "qwen3-vl-plus", "Qwen3.5-397B-A17B",
          "DeepSeek-V4.1-Flash", "DeepSeek-V4-Flash")
ROOT = HARN / "runs_e2_relation"

def done(route):
    d = ROOT / "e2" / route
    return len([p for p in d.glob("*/report.json")]) if d.exists() else 0

for attempt in range(1, 41):
    missing = {r: 38 - done(r) for r in ROUTES if done(r) < 38}
    print(f"[{time.strftime('%H:%M:%S')}] attempt {attempt}: {missing}", flush=True)
    if not missing:
        print("E2 RUN COMPLETE", flush=True)
        break
    procs = []
    for route in missing:
        procs.append(subprocess.Popen(
            [sys.executable, "tools/rendered_relation_experiment.py",
             "--routes", route, "--profiles", *PROFILES,
             "--base-url", os.environ.get("JUDGE_BASE_URL", "https://api.openai.com/v1"),
             "--out", str(ROOT), "--resume"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in procs:
        p.wait()
    time.sleep(5)
else:
    print("E2 RUN INCOMPLETE after 40 attempts", flush=True)
