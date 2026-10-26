"""Drive the blind judge over many run directories, resumably.

Why a driver rather than a shell loop.  The judge is one process per run directory,
and each run takes tens of seconds; a shell loop over ~130 directories needs to
survive interruption and to leave a readable record of what it has already judged.
It also has to skip directories that are already judged, because re-judging would
append a second verdict to a run whose first verdict is already part of the
evidence.

The judge's own checkpoint file (``judge.partial.json``) is what makes a single
directory resumable; this driver only decides *which* directories to hand it.  A
directory counts as done when ``judge.v4.json`` exists, which the judge writes only
after every branch in it has been voted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_BASE_URL = "__REDACTED_BASE_URL__"


def _judged(directory: Path) -> bool:
    final = directory / "judge.v4.json"
    if not final.exists():
        return False
    # A final file with no branches is a partial write, not a verdict.
    try:
        payload = json.loads(final.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(payload.get("branches"))


def plan(roots) -> list:
    todo = []
    for root in roots:
        path = Path(root)
        if not path.exists():
            continue
        for responses in sorted(path.rglob("responses.json")):
            directory = responses.parent
            if _judged(directory):
                continue
            todo.append(directory)
    return todo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--model", default="GLM-5.3-flash")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="JUDGE_API_KEY")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=16000,
                        help="must be generous: the thinking judge spends "
                             "most of the budget on reasoning before it "
                             "emits any verdict")
    parser.add_argument("--limit", type=int, default=0,
                        help="judge at most this many directories; 0 means all")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--progress", default="results_v4/judge_progress.json")
    args = parser.parse_args()

    todo = plan(args.roots)
    print(f"directories awaiting judgement: {len(todo)}")
    if args.plan_only:
        for directory in todo[:40]:
            print(f"  {directory}")
        return 0
    if args.limit:
        todo = todo[:args.limit]

    judge_script = Path(__file__).resolve().parents[1] / "tools" / "judge_v4.py"
    done, failed = [], []
    started = time.monotonic()
    for index, directory in enumerate(todo, start=1):
        command = [
            sys.executable, str(judge_script), str(directory),
            "--base-url", args.base_url, "--model", args.model,
            "--repeats", str(args.repeats),
            "--max-tokens", str(args.max_tokens),
        ]
        began = time.monotonic()
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=1800)
        except subprocess.TimeoutExpired:
            failed.append({"directory": str(directory), "reason": "timeout"})
            print(f"  [{index}/{len(todo)}] TIMEOUT {directory}", flush=True)
            continue
        elapsed = round(time.monotonic() - began, 1)
        if result.returncode == 0 and _judged(directory):
            done.append(str(directory))
            print(f"  [{index}/{len(todo)}] ok {elapsed}s {directory}", flush=True)
        else:
            failed.append({
                "directory": str(directory),
                "returncode": result.returncode,
                "stderr": (result.stderr or "")[-300:],
            })
            print(f"  [{index}/{len(todo)}] FAILED {elapsed}s {directory} "
                  f"rc={result.returncode}", flush=True)
        # Written every few directories so an interruption leaves a usable record.
        if index % 5 == 0 or index == len(todo):
            Path(args.progress).parent.mkdir(parents=True, exist_ok=True)
            Path(args.progress).write_text(json.dumps({
                "done": len(done), "failed": failed,
                "total": len(todo),
                "wall_seconds": round(time.monotonic() - started, 1),
            }, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"judged {len(done)}/{len(todo)}; failures {len(failed)}")
    for entry in failed:
        print(f"  {entry}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
