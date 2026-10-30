#!/usr/bin/env python3
"""
Re-run all 15 PTB-XL FedAvg sessions under the patient-disjoint protocol
into a fresh methodology-v2 cache. CIFAR-10/100 sessions are copied (not re-run).

Protocol tag: ptbxl_patient_disjoint_v2
Cache root:   SINCONF/cache/sinconf2026_methodology_v2/sessions/
Reports:      SINCONF/results/sinconf2026_methodology_v2/

Usage:
  # From SINCONF (or any cwd); PYTHONPATH is set by this script
  python scripts/run_ptbxl_patient_disjoint_sessions.py
  python scripts/run_ptbxl_patient_disjoint_sessions.py --resume
  python scripts/run_ptbxl_patient_disjoint_sessions.py --device cuda --max-records 5000

  # Skip FL and only re-write split/overlap/integrity reports from existing v2 metas
  python scripts/run_ptbxl_patient_disjoint_sessions.py --reports-only

Notes:
  - Does NOT load or copy old ptbxl_* caches from paper2/results/online/sessions.
  - --resume skips PTB-XL configs that already have a complete v2 session (csv+timing+meta).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Paths / PYTHONPATH: monorepo root (fedpbg) + SINCONF (paper2)
# ---------------------------------------------------------------------------
SINCONF_ROOT = Path(__file__).resolve().parents[1]
MONOREPO_ROOT = SINCONF_ROOT.parent
for _p in (str(MONOREPO_ROOT), str(SINCONF_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PROTOCOL = "ptbxl_patient_disjoint_v2"
PROTOCOL_VERSION = "sinconf2026_methodology_v2"

OLD_SESSIONS_DIR = SINCONF_ROOT / "paper2" / "results" / "online" / "sessions"
V2_CACHE_ROOT = SINCONF_ROOT / "cache" / PROTOCOL_VERSION
V2_SESSIONS_DIR = V2_CACHE_ROOT / "sessions"
V2_RESULTS_DIR = SINCONF_ROOT / "results" / PROTOCOL_VERSION

SEEDS = [42, 43, 44, 45, 46]
ALPHAS = [0.1, 0.5, 1.0]
NUM_CLIENTS = 20
DATASET = "ptbxl"
EXPECTED_CIFAR = 30  # 15 cifar10 + 15 cifar100
EXPECTED_PTBXL = 15
EXPECTED_TOTAL = EXPECTED_CIFAR + EXPECTED_PTBXL


def config_id(dataset: str, seed: int, alpha: float, num_clients: int) -> str:
    """Match paper2.src.online.session.config_id (kept local to avoid eager torch import)."""
    return f"{dataset}_s{seed}_a{alpha}_n{num_clients}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run PTB-XL patient-disjoint FedAvg sessions into methodology-v2 cache."
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip PTB-XL configs that already exist in the v2 sessions cache.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: cuda if available else cpu).",
    )
    p.add_argument("--max-records", type=int, default=5000, help="PTB-XL record cap (default 5000).")
    p.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds (default 42-46).")
    p.add_argument(
        "--alphas",
        type=str,
        default=None,
        help="Comma-separated Dirichlet alphas (default 0.1,0.5,1.0).",
    )
    p.add_argument(
        "--reports-only",
        action="store_true",
        help="Do not run FL; only copy CIFAR (if needed) and write reports/integrity from v2 metas.",
    )
    p.add_argument(
        "--sessions-dir",
        type=Path,
        default=None,
        help=f"Override v2 sessions dir (default: {V2_SESSIONS_DIR}).",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=f"Override v2 results/reports dir (default: {V2_RESULTS_DIR}).",
    )
    p.add_argument(
        "--old-sessions-dir",
        type=Path,
        default=None,
        help="Source of CIFAR sessions only (default: paper2/results/online/sessions).",
    )
    return p.parse_args()


def assert_not_legacy_ptbxl_eval_source(sessions_dir: Path) -> None:
    """Refuse to treat the old paper2 online sessions dir as the v2 PTB-XL cache."""
    if sessions_dir.resolve() == OLD_SESSIONS_DIR.resolve():
        raise RuntimeError(
            "Refusing to use paper2/results/online/sessions as the methodology-v2 PTB-XL "
            "cache. Re-run into cache/sinconf2026_methodology_v2/sessions/ instead."
        )


def session_paths(sessions_dir: Path, cid: str) -> Tuple[Path, Path, Path]:
    csv_p = sessions_dir / f"{cid}_client_level.csv"
    timing_p = sessions_dir / f"{cid}_timing.json"
    meta_p = sessions_dir / f"{cid}_meta.json"
    return csv_p, timing_p, meta_p


def session_complete(sessions_dir: Path, cid: str) -> bool:
    csv_p, timing_p, meta_p = session_paths(sessions_dir, cid)
    return csv_p.exists() and timing_p.exists() and meta_p.exists()


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_session_meta(session, out_dir: Path) -> Path:
    """Persist session.meta (includes patient_split) next to csv/timing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(session.meta or {})
    meta.setdefault("protocol_version", PROTOCOL_VERSION)
    meta["protocol"] = PROTOCOL
    ps = meta.get("patient_split")
    if isinstance(ps, dict):
        ps = dict(ps)
        ps["protocol"] = PROTOCOL
        meta["patient_split"] = ps
    meta_path = out_dir / f"{session.config_id}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=_json_default), encoding="utf-8")
    return meta_path


def copy_cifar_sessions(src_dir: Path, dst_dir: Path) -> int:
    """Copy cifar10_* and cifar100_* session artifacts; never copy ptbxl_*."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"CIFAR source sessions dir not found: {src_dir}")

    copied = 0
    for path in sorted(src_dir.iterdir()):
        name = path.name
        if name.startswith("ptbxl"):
            print(f"  Skip legacy PTB-XL (refuse copy): {name}", flush=True)
            continue
        if not (name.startswith("cifar10_") or name.startswith("cifar100_")):
            continue
        if not (
            name.endswith("_client_level.csv")
            or name.endswith("_timing.json")
            or name.endswith("_meta.json")
        ):
            continue
        target = dst_dir / name
        if not target.exists() or target.stat().st_mtime < path.stat().st_mtime:
            shutil.copy2(path, target)
            copied += 1
            print(f"  Copied CIFAR: {name}", flush=True)
        else:
            print(f"  CIFAR exists: {name}", flush=True)
    return copied


def _pairwise_overlaps(groups: Sequence[Set[str]]) -> int:
    n = 0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            n += len(groups[i] & groups[j])
    return n


def analyze_patient_split(meta: Dict[str, Any], config_id_str: str) -> Dict[str, Any]:
    """Return overlap counts and protocol info from a saved meta dict."""
    ps = meta.get("patient_split") or {}
    protocol = ps.get("protocol") or meta.get("protocol")
    global_split = ps.get("global_split_patients") or {}
    client_patients = ps.get("client_patients") or {}

    train = set(map(str, global_split.get("train", [])))
    val = set(map(str, global_split.get("val", [])))
    test = set(map(str, global_split.get("test", [])))
    global_overlap = (
        len(train & val) + len(train & test) + len(val & test)
    )

    client_sets = [set(map(str, pids)) for pids in client_patients.values()]
    client_overlap = _pairwise_overlaps(client_sets)

    # Within-client: train/val/test derived from global & client - check global only here;
    # cross-client is the main federated leakage vector.
    return {
        "config_id": config_id_str,
        "protocol": protocol,
        "n_patients": ps.get("n_patients"),
        "n_records": ps.get("n_records"),
        "global_train_val_test_overlap_patients": global_overlap,
        "cross_client_overlap_patients": client_overlap,
        "split_report_rows": ps.get("split_report_rows") or [],
        "ok": (
            protocol == PROTOCOL
            and global_overlap == 0
            and client_overlap == 0
            and bool(global_split)
            and bool(client_patients)
        ),
    }


def write_split_and_overlap_reports(
    sessions_dir: Path,
    results_dir: Path,
    seeds: Sequence[int],
    alphas: Sequence[float],
) -> Tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    split_rows: List[Dict[str, Any]] = []
    overlap_lines: List[str] = [
        f"PTB-XL patient-disjoint overlap report ({PROTOCOL})",
        f"sessions_dir: {sessions_dir}",
        "",
    ]
    total_global = 0
    total_client = 0
    n_checked = 0

    for seed, alpha in product(seeds, alphas):
        cid = config_id(DATASET, seed, alpha, NUM_CLIENTS)
        _, _, meta_p = session_paths(sessions_dir, cid)
        if not meta_p.exists():
            overlap_lines.append(f"[MISSING META] {cid}")
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        info = analyze_patient_split(meta, cid)
        n_checked += 1
        total_global += int(info["global_train_val_test_overlap_patients"])
        total_client += int(info["cross_client_overlap_patients"])
        for row in info["split_report_rows"]:
            r = dict(row)
            r["config_id"] = cid
            r["alpha"] = alpha
            r.setdefault("seed", seed)
            split_rows.append(r)
        status = "OK" if info["ok"] else "FAIL"
        overlap_lines.append(
            f"[{status}] {cid} protocol={info['protocol']} "
            f"global_overlap={info['global_train_val_test_overlap_patients']} "
            f"cross_client_overlap={info['cross_client_overlap_patients']}"
        )

    overlap_lines.extend(
        [
            "",
            f"sessions_checked: {n_checked}",
            f"total_global_train_val_test_overlap_patients: {total_global}",
            f"total_cross_client_overlap_patients: {total_client}",
            f"zero_overlaps: {total_global == 0 and total_client == 0}",
        ]
    )
    if total_global != 0 or total_client != 0:
        overlap_lines.append("ERROR: non-zero patient overlaps detected.")

    split_csv = results_dir / "ptbxl_patient_split_report.csv"
    overlap_txt = results_dir / "ptbxl_patient_overlap_report.txt"
    import pandas as pd

    pd.DataFrame(split_rows).to_csv(split_csv, index=False)
    overlap_txt.write_text("\n".join(overlap_lines) + "\n", encoding="utf-8")
    print(f"  Wrote {split_csv}", flush=True)
    print(f"  Wrote {overlap_txt}", flush=True)
    if total_global != 0 or total_client != 0:
        raise RuntimeError("Patient overlap report is non-zero; aborting.")
    return split_csv, overlap_txt


def verify_session_counts(sessions_dir: Path) -> Dict[str, int]:
    csvs = list(sessions_dir.glob("*_client_level.csv"))
    counts = {"cifar10": 0, "cifar100": 0, "ptbxl": 0, "other": 0}
    for p in csvs:
        name = p.name
        if name.startswith("cifar10_"):
            counts["cifar10"] += 1
        elif name.startswith("cifar100_"):
            counts["cifar100"] += 1
        elif name.startswith("ptbxl"):
            counts["ptbxl"] += 1
        else:
            counts["other"] += 1
    counts["total"] = len(csvs)
    return counts


def integrity_check_ptbxl_metas(
    sessions_dir: Path,
    results_dir: Path,
    seeds: Sequence[int],
    alphas: Sequence[float],
) -> Path:
    """Load each PTB-XL session meta and confirm protocol tag."""
    assert_not_legacy_ptbxl_eval_source(sessions_dir)
    lines = [
        f"PTB-XL session meta integrity ({PROTOCOL})",
        f"sessions_dir: {sessions_dir}",
        "",
    ]
    failures: List[str] = []
    for seed, alpha in product(seeds, alphas):
        cid = config_id(DATASET, seed, alpha, NUM_CLIENTS)
        csv_p, timing_p, meta_p = session_paths(sessions_dir, cid)
        # Refuse accidental evaluation against legacy tree
        for p in (csv_p, timing_p, meta_p):
            if OLD_SESSIONS_DIR.resolve() in p.resolve().parents or p.resolve().parent == OLD_SESSIONS_DIR.resolve():
                if p.name.startswith("ptbxl"):
                    raise RuntimeError(f"Refusing to load legacy PTB-XL artifact for eval: {p}")

        if not meta_p.exists():
            msg = f"MISSING meta: {cid}"
            failures.append(msg)
            lines.append(f"[FAIL] {msg}")
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        ps = meta.get("patient_split") or {}
        protocol = ps.get("protocol") or meta.get("protocol")
        proto_ver = meta.get("protocol_version")
        ok = protocol == PROTOCOL and proto_ver == PROTOCOL_VERSION
        if not ok:
            msg = (
                f"{cid}: protocol={protocol!r} protocol_version={proto_ver!r} "
                f"(expected {PROTOCOL!r} / {PROTOCOL_VERSION!r})"
            )
            failures.append(msg)
            lines.append(f"[FAIL] {msg}")
        else:
            lines.append(
                f"[OK] {cid} protocol={protocol} protocol_version={proto_ver} "
                f"csv={csv_p.exists()} timing={timing_p.exists()}"
            )

    lines.append("")
    lines.append(f"failures: {len(failures)}")
    out = results_dir / "ptbxl_meta_integrity.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Wrote {out}", flush=True)
    if failures:
        raise RuntimeError(f"PTB-XL meta integrity failed ({len(failures)} issues). See {out}")
    return out


def run_ptbxl_grid(
    sessions_dir: Path,
    seeds: Sequence[int],
    alphas: Sequence[float],
    device: Any,
    resume: bool,
    max_records: int,
) -> List[str]:
    import torch
    from paper2.src.online.session import run_online_fl_session, save_session

    assert_not_legacy_ptbxl_eval_source(sessions_dir)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    grid = list(product(seeds, alphas))
    done_ids: List[str] = []
    t_all = time.perf_counter()

    for i, (seed, alpha) in enumerate(grid, start=1):
        cid = config_id(DATASET, seed, alpha, NUM_CLIENTS)
        csv_p, _, meta_p = session_paths(sessions_dir, cid)

        # Never resume from legacy PTB-XL paths
        legacy_csv = OLD_SESSIONS_DIR / f"{cid}_client_level.csv"
        if resume and csv_p.exists() and meta_p.exists():
            if csv_p.resolve() == legacy_csv.resolve():
                raise RuntimeError(f"Refusing to resume from legacy PTB-XL cache: {legacy_csv}")
            print(f"[{i}/{len(grid)}] Resume: {cid}", flush=True)
            done_ids.append(cid)
            continue

        print(f"[{i}/{len(grid)}] Running: {cid} (device={device}, max_records={max_records}) ...", flush=True)
        t0 = time.perf_counter()
        session = run_online_fl_session(
            dataset_name=DATASET,
            seed=seed,
            alpha=alpha,
            num_clients=NUM_CLIENTS,
            global_rounds=5,
            local_epochs=3,
            device=device,
            quick=False,
            max_records=max_records,
        )
        # Guarantee protocol tags on returned meta
        session.meta = dict(session.meta or {})
        session.meta["protocol"] = PROTOCOL
        session.meta["protocol_version"] = PROTOCOL_VERSION
        if isinstance(session.meta.get("patient_split"), dict):
            session.meta["patient_split"]["protocol"] = PROTOCOL

        save_session(session, sessions_dir)
        save_session_meta(session, sessions_dir)
        elapsed = time.perf_counter() - t0
        total_sec = float(session.timing.get("total_sec", elapsed))
        print(
            f"    Done {cid} in {elapsed:.1f}s wall "
            f"(session.total_sec={total_sec:.1f}s) -> {csv_p.name}",
            flush=True,
        )
        done_ids.append(cid)

    print(f"PTB-XL grid wall time: {time.perf_counter() - t_all:.1f}s", flush=True)
    return done_ids


def main() -> int:
    args = parse_args()
    sessions_dir = (args.sessions_dir or V2_SESSIONS_DIR).resolve()
    results_dir = (args.results_dir or V2_RESULTS_DIR).resolve()
    old_sessions = (args.old_sessions_dir or OLD_SESSIONS_DIR).resolve()

    seeds = [int(x) for x in (args.seeds or ",".join(map(str, SEEDS))).split(",") if x.strip()]
    alphas = [float(x) for x in (args.alphas or ",".join(map(str, ALPHAS))).split(",") if x.strip()]

    if args.reports_only:
        device = None
    else:
        import torch

        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    assert_not_legacy_ptbxl_eval_source(sessions_dir)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print("SINCONF methodology v2 - PTB-XL patient-disjoint sessions", flush=True)
    print(f"  protocol:        {PROTOCOL}", flush=True)
    print(f"  sessions_dir:    {sessions_dir}", flush=True)
    print(f"  results_dir:     {results_dir}", flush=True)
    print(f"  cifar_source:    {old_sessions}", flush=True)
    print(f"  seeds:           {seeds}", flush=True)
    print(f"  alphas:          {alphas}", flush=True)
    print(f"  device:          {device}", flush=True)
    print(f"  max_records:     {args.max_records}", flush=True)
    print(f"  resume:          {args.resume}", flush=True)
    print(f"  PYTHONPATH roots:{MONOREPO_ROOT} | {SINCONF_ROOT}", flush=True)
    print("=" * 72, flush=True)

    print("\n[1/4] Copy CIFAR-10/100 sessions into v2 cache (no re-run)...", flush=True)
    t0 = time.perf_counter()
    n_copied = copy_cifar_sessions(old_sessions, sessions_dir)
    print(f"  Copy step finished in {time.perf_counter() - t0:.1f}s ({n_copied} files updated).", flush=True)

    if not args.reports_only:
        print(f"\n[2/4] Run PTB-XL FedAvg grid ({len(seeds) * len(alphas)} configs)...", flush=True)
        run_ptbxl_grid(
            sessions_dir=sessions_dir,
            seeds=seeds,
            alphas=alphas,
            device=device,
            resume=args.resume,
            max_records=args.max_records,
        )
    else:
        print("\n[2/4] --reports-only: skipping PTB-XL FL runs.", flush=True)

    print("\n[3/4] Patient split + overlap reports...", flush=True)
    write_split_and_overlap_reports(sessions_dir, results_dir, seeds, alphas)

    print("\n[4/4] Verify session counts + PTB-XL meta integrity...", flush=True)
    counts = verify_session_counts(sessions_dir)
    print(
        f"  Counts: total={counts['total']} "
        f"(cifar10={counts['cifar10']}, cifar100={counts['cifar100']}, "
        f"ptbxl={counts['ptbxl']}, other={counts['other']})",
        flush=True,
    )
    cifar_n = counts["cifar10"] + counts["cifar100"]
    if cifar_n != EXPECTED_CIFAR or counts["ptbxl"] != EXPECTED_PTBXL or counts["total"] != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} sessions "
            f"({EXPECTED_CIFAR} cifar + {EXPECTED_PTBXL} ptbxl); "
            f"found total={counts['total']} cifar={cifar_n} ptbxl={counts['ptbxl']}"
        )
    print(f"  OK: {EXPECTED_TOTAL} sessions present in v2 cache.", flush=True)
    integrity_check_ptbxl_metas(sessions_dir, results_dir, seeds, alphas)

    print("\n" + "=" * 72, flush=True)
    print("PTB-XL patient-disjoint sessions complete.", flush=True)
    print(f"  Sessions: {sessions_dir}", flush=True)
    print(f"  Reports:  {results_dir}", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", flush=True)
        raise
