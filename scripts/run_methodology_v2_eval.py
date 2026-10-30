#!/usr/bin/env python3
"""
Run SINCONF methodology v2 evaluation (grouped CV + session-level stats).

Does not fabricate results. Reads online session CSVs, writes all tables/figures
under --out-dir.

Usage:
  python scripts/run_methodology_v2_eval.py \\
    --sessions-dir paper2/results/online/sessions \\
    --out-dir results/sinconf2026_methodology_v2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# PYTHONPATH: prefer SINCONF (paper2) over parent monorepo; fedpbg from monorepo/artifact
SINCONF_ROOT = Path(__file__).resolve().parents[1]
MONOREPO_ROOT = SINCONF_ROOT.parent
# insert(0) reverses order — put lowest-priority paths first so SINCONF wins
for p in (
    SINCONF_ROOT / "tmlr" / "artifact",
    MONOREPO_ROOT,
    SINCONF_ROOT,
):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SINCONF methodology v2 evaluation")
    p.add_argument(
        "--sessions-dir",
        type=str,
        default=str(SINCONF_ROOT / "paper2" / "results" / "online" / "sessions"),
        help="Directory of *_client_level.csv FedAvg sessions",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(SINCONF_ROOT / "results" / "sinconf2026_methodology_v2"),
        help="Output root for tables, figures, and reports",
    )
    p.add_argument("--n-splits", type=int, default=5, help="GroupKFold splits")
    p.add_argument("--bootstrap-B", type=int, default=10000, help="Cluster bootstrap resamples")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip PNG generation / submission copy",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sessions_dir = Path(args.sessions_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from paper2.src.methodology_v2 import PROTOCOL
    from paper2.src.methodology_v2.build_rows import build_policy_learning_rows
    from paper2.src.methodology_v2.cluster_stats import run_cluster_stats
    from paper2.src.methodology_v2.comparison import run_comparison
    from paper2.src.methodology_v2.figures import generate_figures
    from paper2.src.methodology_v2.grouped_eval import run_grouped_evaluation

    t0 = time.perf_counter()
    print("=" * 72, flush=True)
    print(f"SINCONF methodology v2  [{PROTOCOL}]", flush=True)
    print(f"  sessions-dir: {sessions_dir}", flush=True)
    print(f"  out-dir:      {out_dir}", flush=True)
    print("=" * 72, flush=True)

    print("\n[1/5] Building policy-learning rows from session CSVs...", flush=True)
    rows = build_policy_learning_rows(sessions_dir)
    rows_path = out_dir / "policy_learning_rows.csv"
    rows.to_csv(rows_path, index=False)
    n_sessions = rows["session_id"].nunique()
    n_clients = rows["client_instance_id"].nunique()
    print(
        f"  rows={len(rows)}  sessions={n_sessions}  client_instances={n_clients}  "
        f"lambdas={sorted(rows['lambda'].unique())}",
        flush=True,
    )
    print(f"  wrote {rows_path}", flush=True)

    print("\n[2/5] Grouped evaluation (GroupKFold + LOO seed/dataset)...", flush=True)
    eval_out = run_grouped_evaluation(
        rows,
        out_dir=out_dir,
        n_splits=args.n_splits,
        random_state=args.seed,
    )
    gkf = eval_out["group_kfold"]
    oof = eval_out["oof"]
    print("  GroupKFold policy metrics:", flush=True)
    for policy in ("greedy", "fedpba", "best_uniform", "context_only", "majority"):
        m = gkf.get(policy, {})
        if not m:
            continue
        print(
            f"    {policy:14s}  regret={m.get('mean_regret', float('nan')):.6f}  "
            f"recovery={m.get('oracle_utility_recovery_pct', float('nan')):.2f}%  "
            f"acc={m.get('action_accuracy', float('nan')):.4f}",
            flush=True,
        )
    print(f"  split integrity folds: {len(eval_out['integrity'])}", flush=True)

    print("\n[3/5] Session-level cluster stats (B={})...".format(args.bootstrap_B), flush=True)
    cluster = run_cluster_stats(
        oof,
        out_dir=out_dir,
        n_boot=args.bootstrap_B,
        seed=args.seed,
    )
    holm = cluster.get("holm_corrected_primary_tests")
    if holm is not None and len(holm):
        print("  Primary comparisons (Holm):", flush=True)
        for _, r in holm.iterrows():
            print(
                f"    {r['comparison']}: dU={r['mean_diff']:.6f}  "
                f"p={r['wilcoxon_p']:.4g}  Holm_p={r['holm_adjusted_p']:.4g}  "
                f"dz={r['cohens_dz']:.4f}  n={int(r['n_sessions'])}",
                flush=True,
            )

    print("\n[4/5] Old vs new comparison CSV...", flush=True)
    tables_dir = SINCONF_ROOT / "paper2" / "tables"
    cmp_df = run_comparison(
        gkf,
        cluster,
        out_dir=out_dir,
        oof=oof,
        tables_dir=tables_dir,
    )
    print(f"  comparison rows: {len(cmp_df)}", flush=True)
    print(f"  wrote {out_dir / 'methodology_revision_comparison.csv'}", flush=True)

    fig_paths = {}
    if not args.skip_figures:
        print("\n[5/5] Regenerating figures...", flush=True)
        fig_paths = generate_figures(cluster, out_dir=out_dir, also_copy_to_submission=True)
        for k, p in fig_paths.items():
            print(f"  {k}: {p}", flush=True)
    else:
        print("\n[5/5] Skipping figures (--skip-figures)", flush=True)

    # Summary JSON (computed values only)
    summary = {
        "protocol": PROTOCOL,
        "n_rows": int(len(rows)),
        "n_sessions": int(n_sessions),
        "n_client_instances": int(n_clients),
        "group_kfold": gkf,
        "elapsed_sec": float(time.perf_counter() - t0),
        "figures": {k: str(v) for k, v in fig_paths.items()},
    }
    with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 72, flush=True)
    print("DONE", flush=True)
    print(f"  elapsed: {summary['elapsed_sec']:.1f}s", flush=True)
    print(f"  outputs: {out_dir}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
