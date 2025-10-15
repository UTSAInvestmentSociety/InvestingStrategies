#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run strategy scripts sequentially.

This runner will execute, in order:
  1) strategy_01_momentum.py
  2) strategy_mean_variance_opt.py
  3) strategy_value_screens.py

It streams each script's output to the console AND saves it to individual log files
under ./logs/, stopping on the first non-zero exit code (unless --continue-on-error).

Usage (from the folder containing the strategy scripts):
    python run_all_strategies.py
Optional flags:
    --continue-on-error    keep going even if a script fails
    --timeout SEC          kill a script if it runs longer than SEC seconds
    --python PYEXEC        path to a specific python executable (default: current)
"""

import argparse
import datetime as dt
import os
import signal
import sys
import time
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT

SCRIPTS_IN_ORDER = [
    "strategy_01_momentum.py",
    "strategy_mean_variance_opt.py",
    "strategy_value_screens.py",
]

def run_script(py_exec: str, script_path: Path, log_dir: Path, timeout: float | None) -> int:
    """Run a script, streaming stdout/stderr to console and a log file. Returns exit code."""
    script_path = script_path.resolve()
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return 127

    started = dt.datetime.now()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{script_path.stem}_{started.strftime('%Y%m%d_%H%M%S')}.log"

    print(f"\n=== Running: {script_path.name} ===")
    print(f"[INFO] Start time: {started.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Logging to: {log_file}")

    with log_file.open("w", encoding="utf-8") as lf:
        lf.write(f"# {script_path.name} | started {started.isoformat()}\n\n")
        # Use unbuffered (-u) to get real-time output
        proc = Popen([py_exec, "-u", str(script_path)], cwd=str(script_path.parent),
                     stdout=PIPE, stderr=STDOUT, text=True, bufsize=1)

        try:
            t0 = time.time()
            while True:
                line = proc.stdout.readline()
                if line == "" and proc.poll() is not None:
                    break
                if line:
                    print(line, end="")       # stream to console
                    lf.write(line)            # write to log
                    lf.flush()
                if timeout is not None and (time.time() - t0) > timeout:
                    print(f"\n[WARN] Timeout reached ({timeout}s). Terminating {script_path.name}...")
                    # Try graceful, then force
                    try:
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except Exception:
                            proc.kill()
                    except Exception:
                        proc.kill()
                    return 124  # common timeout code
            rc = proc.returncode
        except KeyboardInterrupt:
            print("\n[INFO] KeyboardInterrupt received. Terminating child process...")
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
            return 130  # SIGINT
        finally:
            ended = dt.datetime.now()
            lf.write(f"\n# finished {ended.isoformat()} | elapsed {ended - started}\n")

    elapsed = dt.datetime.now() - started
    print(f"[INFO] Finished: {script_path.name} | exit={rc} | elapsed={elapsed}\n")
    return rc

def main():
    parser = argparse.ArgumentParser(description="Run strategy scripts sequentially.")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="continue to next script even if one fails")
    parser.add_argument("--timeout", type=float, default=None,
                        help="per-script timeout in seconds (default: no timeout)")
    parser.add_argument("--python", dest="pyexec", default=sys.executable,
                        help="python executable to use (default: current interpreter)")
    parser.add_argument("--base-dir", default=".", help="directory containing the scripts (default: .)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    log_dir = base_dir / "logs"

    print(f"[INFO] Using Python: {args.pyexec}")
    print(f"[INFO] Base directory: {base_dir}")
    print(f"[INFO] Scripts: {', '.join(SCRIPTS_IN_ORDER)}")

    overall = []
    for name in SCRIPTS_IN_ORDER:
        script_path = base_dir / name
        rc = run_script(args.pyexec, script_path, log_dir, args.timeout)
        overall.append((name, rc))
        if rc != 0 and not args.continue_on_error:
            print(f"[ERROR] {name} failed with exit code {rc}. Halting.")
            break

    # Summary
    print("\n==== Summary ====")
    for name, rc in overall:
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"{name:>28} : {status}")

    # Non-zero exit if any failed
    sys.exit(0 if all(rc == 0 for _, rc in overall) else 1)

if __name__ == "__main__":
    main()
