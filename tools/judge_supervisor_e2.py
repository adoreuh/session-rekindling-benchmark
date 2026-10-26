import os, subprocess, sys, time
from pathlib import Path
HARN = Path(__file__).resolve().parents[1]
os.chdir(HARN)
env = dict(os.environ)
env.setdefault("JUDGE_API_KEY", os.environ.get("JUDGE_API_KEY", ""))
TARGETS = [(HARN / "runs_e2_relation", HARN / "results_v4" / "e2_judge_report.json")]

def shard_dirs(root):
    out = []
    for container in root.iterdir():
        if not container.is_dir():
            continue
        for route_dir in container.iterdir():
            if not route_dir.is_dir():
                continue
            cells = [d for d in route_dir.iterdir() if d.is_dir()]
            if any((c / "report.json").exists() and not (c / "judge.expiry.json").exists()
                   for c in cells):
                out.append(route_dir)
    return out

procs = {}
while True:
    procs = {k: p for k, p in procs.items() if p.poll() is None}
    shards = [(root, report, d) for root, report in TARGETS for d in shard_dirs(root)]
    print(f"[{time.strftime('%H:%M:%S')}] dirty={len(shards)} live={len(procs)}", flush=True)
    for root, report, d in shards:
        if str(d) in procs:
            continue
        procs[str(d)] = subprocess.Popen(
            [sys.executable, "tools/expiry_judge.py", str(d),
             "--base-url", os.environ.get("JUDGE_BASE_URL", "https://api.openai.com/v1"),
             "--api-key-env", "JUDGE_API_KEY", "--report", str(report)],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not shards:
        for root, report in TARGETS:
            subprocess.run([sys.executable, "tools/expiry_judge.py", str(root),
                            "--base-url", os.environ.get("JUDGE_BASE_URL", "https://api.openai.com/v1"),
                            "--api-key-env", "JUDGE_API_KEY", "--report", str(report)],
                           env=env, capture_output=True)
        print("ALL DONE", flush=True)
        break
    time.sleep(30)
