#!/usr/bin/env python3
"""
Orchestrate SINCONF methodology v2: PTB-XL patient-disjoint sessions, then evaluation.

Does NOT update the manuscript. Exits non-zero if any step fails.

Usage:
  python scripts/run_methodology_v2_pipeline.py
  python scripts/run_methodology_v2_pipeline.py --resume
  python scripts/run_methodology_v2_pipeline.py --skip-sessions   # eval only
  python scripts/run_methodology_v2_pipeline.py --skip-eval       # sessions only
  python scripts/run_methodology_v2_pipeline.py --device cuda --max-records 5000

Steps:
  1. scripts/run_ptbxl_patient_disjoint_sessions.py
  2. scripts/run_methodology_v2_eval.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SINCONF_ROOT = Path(__file__).resolve().parents[1]
MONOREPO_ROOT = SINCONF_ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

SESSIONS_SCRIPT = SCRIPTS_DIR / "run_ptbxl_patient_disjoint_sessions.py"
EVAL_SCRIPT = SCRIPTS_DIR / "run_methodology_v2_eval.py"

PROTOCOL_VERSION = "sinconf2026_methodology_v2"
DEFAULT_SESSIONS_DIR = SINCONF_ROOT / "cache" / PROTOCOL_VERSION / "sessions"
DEFAULT_OUT_DIR = SINCONF_ROOT / "results" / PROTOCOL_VERSION


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run methodology v2 pipeline (PTB-XL sessions -> evaluation)."
    )
    p.add_argument("--resume", action="store_true", help="Pass --resume to the sessions script.")
    p.add_argument("--device", type=str, default=None, help="Torch device for PTB-XL sessions.")
    p.add_argument("--max-records", type=int, default=5000, help="PTB-XL max_records (default 5000).")
    p.add_argument(
        "--sessions-dir",
        type=Path,
        default=None,
        help=f"V2 sessions directory (default: {DEFAULT_SESSIONS_DIR}).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Evaluation output directory (default: {DEFAULT_OUT_DIR}).",
    )
    p.add_argument("--skip-sessions", action="store_true", help="Skip PTB-XL session regeneration.")
    p.add_argument("--skip-eval", action="store_true", help="Skip methodology v2 evaluation.")
    p.add_argument(
        "--reports-only",
        action="store_true",
        help="Pass --reports-only to the sessions script (no FL).",
    )
    return p.parse_args()


def _python_env() -> dict:
    import os

    env = os.environ.copy()
    parts = [str(MONOREPO_ROOT), str(SINCONF_ROOT)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def run_step(label: str, cmd: list[str]) -> None:
    print("=" * 72, flush=True)
    print(f"STEP: {label}", flush=True)
    print(f"CMD:  {' '.join(cmd)}", flush=True)
    print("=" * 72, flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(SINCONF_ROOT), env=_python_env())
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(
            f"\nFAILED: {label} (exit={proc.returncode}, wall={elapsed:.1f}s)",
            flush=True,
        )
        raise SystemExit(proc.returncode)
    print(f"\nOK: {label} finished in {elapsed:.1f}s", flush=True)


def main() -> int:
    args = parse_args()
    sessions_dir = (args.sessions_dir or DEFAULT_SESSIONS_DIR).resolve()
    out_dir = (args.out_dir or DEFAULT_OUT_DIR).resolve()
    py = sys.executable

    if not args.skip_sessions and not SESSIONS_SCRIPT.is_file():
        print(f"ERROR: missing sessions script: {SESSIONS_SCRIPT}", flush=True)
        return 1
    if not args.skip_eval and not EVAL_SCRIPT.is_file():
        print(f"ERROR: missing eval script: {EVAL_SCRIPT}", flush=True)
        return 1
    if args.skip_sessions and args.skip_eval:
        print("ERROR: nothing to do (--skip-sessions and --skip-eval).", flush=True)
        return 1

    print("SINCONF methodology v2 pipeline", flush=True)
    print(f"  SINCONF_ROOT:  {SINCONF_ROOT}", flush=True)
    print(f"  MONOREPO_ROOT: {MONOREPO_ROOT}", flush=True)
    print(f"  sessions_dir:  {sessions_dir}", flush=True)
    print(f"  out_dir:       {out_dir}", flush=True)
    print("  manuscript:    not updated by this pipeline", flush=True)

    t_pipeline = time.perf_counter()

    if not args.skip_sessions:
        cmd = [py, str(SESSIONS_SCRIPT), "--sessions-dir", str(sessions_dir), "--results-dir", str(out_dir)]
        if args.resume:
            cmd.append("--resume")
        if args.reports_only:
            cmd.append("--reports-only")
        if args.device:
            cmd.extend(["--device", args.device])
        if args.max_records is not None:
            cmd.extend(["--max-records", str(args.max_records)])
        run_step("PTB-XL patient-disjoint sessions", cmd)
    else:
        print("Skipping sessions step (--skip-sessions).", flush=True)

    if not args.skip_eval:
        cmd = [
            py,
            str(EVAL_SCRIPT),
            "--sessions-dir",
            str(sessions_dir),
            "--out-dir",
            str(out_dir),
        ]
        run_step("Methodology v2 evaluation", cmd)
    else:
        print("Skipping eval step (--skip-eval).", flush=True)

    print("=" * 72, flush=True)
    print(
        f"Pipeline complete in {time.perf_counter() - t_pipeline:.1f}s "
        "(manuscript not modified).",
        flush=True,
    )
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
