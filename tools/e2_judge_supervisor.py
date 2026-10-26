"""Judge the E2 grid route-by-route in parallel, retrying until coverage is full."""
import os, subprocess, sys, time
from pathlib import Path

HARN = Path(__file__).resolve().parents[1]
os.chdir(HARN)
env = dict(os.environ)
env.setdefault("JUDGE_API_KEY", os.environ.get("JUDGE_API_KEY", ""))
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "https://api.openai.com/v1")
ROUTES = ("hy3", "qwen3-vl-plus", "Qwen3.5-397B-A17B",
          "DeepSeek-V4.1-Flash", "DeepSeek-V4-Flash")
ROOT = HARN / "runs_e2_relation"

def judged(route):
    d = ROOT / "e2" / route
    return len(list(d.rglob("judge.expiry.json"))) if d.exists() else 0

def cells(route):
    d = ROOT / "e2" / route
    return len(list(d.rglob("report.json"))) if d.exists() else 0

for attempt in range(1, 61):
    pending = {r: cells(r) - judged(r) for r in ROUTES if judged(r) < cells(r)}
    print(f"[{time.strftime('%H:%M:%S')}] attempt {attempt}: pending={pending} "
          f"total={sum(judged(r) for r in ROUTES)}/190", flush=True)
    if not pending:
        print("E2 JUDGE COMPLETE", flush=True)
        break
    procs = []
    for route in pending:
        procs.append(subprocess.Popen(
            [sys.executable, "tools/expiry_judge.py", str(ROOT / "e2" / route),
             "--base-url", JUDGE_BASE_URL,
             "--api-key-env", "JUDGE_API_KEY",
             "--report", str(HARN / "results_v4" / f"e2_judge_{route}.json")],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in procs:
        p.wait()
    time.sleep(3)
else:
    print("E2 JUDGE INCOMPLETE", flush=True)

# final whole-root pass so the consolidated report covers all 190 cells
subprocess.run([sys.executable, "tools/expiry_judge.py", str(ROOT),
                "--base-url", JUDGE_BASE_URL,
                "--api-key-env", "JUDGE_API_KEY",
                "--report", str(HARN / "results_v4" / "e2_judge_report.json")],
               env=env, capture_output=True)
print("wrote consolidated e2_judge_report.json", flush=True)
