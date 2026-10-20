"""CLI: ``python -m runstore {ingest,status,doctor,prune,ingest-derived} --runs <runs-root>``."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import doctor, ingest, ingest_derived, prune, prune_derived, status


def main() -> None:
    parser = argparse.ArgumentParser(prog="runstore")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="fold run.json shards into one parquet table")
    p_ingest.add_argument("--runs", required=True, help="runs root, e.g. results/<campaign>/runs")
    p_ingest.add_argument("--out", default=None, help="default: <runs>/../db/runs.parquet")

    p_status = sub.add_parser("status", help="done/failed/pending counts per group x task x method")
    p_status.add_argument("--runs", required=True)

    p_doctor = sub.add_parser(
        "doctor", help="quarantine corrupt shards/specs (-> *.corrupt, cell reads as "
                       "pending again) and remove leaked atomic-write temporaries")
    p_doctor.add_argument("--runs", required=True)
    p_doctor.add_argument("--min-age", type=float, default=3600.0,
                          help="only remove temporaries older than this many seconds "
                               "(guards a live writer; 0 only when no workers run)")
    p_doctor.add_argument("--dry-run", action="store_true", help="report without touching files")

    p_prune = sub.add_parser(
        "prune", help="after doctor + ingest: drop per-run files the shard already "
                      "carries (keeps run.json; checkpointed runs also keep spec.json)")
    p_prune.add_argument("--runs", required=True)
    p_prune.add_argument("--dry-run", action="store_true", help="report without touching files")

    p_derived = sub.add_parser(
        "ingest-derived", help="consolidate one versioned derived analysis "
                               "(derived/<analysis>_v<version>/metrics.json) into "
                               "db/<analysis>_v<version>.parquet, joined by run_id")
    p_derived.add_argument("--runs", required=True)
    p_derived.add_argument("--analysis", required=True)
    p_derived.add_argument("--version", type=int, required=True)
    p_derived.add_argument("--out", default=None,
                           help="default: <runs>/../db/<analysis>_v<version>.parquet")

    p_retire = sub.add_parser(
        "prune-derived", help="retire one derived analysis version: remove its "
                              "derived/<analysis>_v<version>/ dirs (record and "
                              "model state untouched)")
    p_retire.add_argument("--runs", required=True)
    p_retire.add_argument("--analysis", required=True)
    p_retire.add_argument("--version", type=int, required=True)
    p_retire.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.cmd == "ingest":
        out = Path(args.out) if args.out else Path(args.runs).parent / "db" / "runs.parquet"
        n = ingest(args.runs, out)
        print(f"{n} run(s) -> {out}" if n else f"no run shards under {args.runs}")
    elif args.cmd == "status":
        print(status(args.runs))
    elif args.cmd == "doctor":
        report = doctor(args.runs, min_age_s=args.min_age, apply=not args.dry_run)
        verb = "would quarantine" if args.dry_run else "quarantined"
        for item in report["quarantined"]:
            print(f"{verb} {item['path']} -> {item['moved_to']}: {item['reason']}")
        print(f"{report['n_cells']} cell(s): {report['n_ok']} ok, "
              f"{len(report['quarantined'])} record(s) {verb}, "
              f"{report['temps_removed']} leaked temp file(s) "
              f"{'found' if args.dry_run else 'removed'}")
    elif args.cmd == "prune":
        report = prune(args.runs, apply=not args.dry_run)
        verb = "would remove" if args.dry_run else "removed"
        print(f"{report['n_dirs']} shard dir(s): {verb} {report['n_files_removed']} "
              f"file(s) / {report['bytes_freed'] / 1e6:.1f} MB from "
              f"{report['n_pruned']} dir(s); {report['n_skipped_invalid']} invalid "
              "shard(s) skipped")
    elif args.cmd == "ingest-derived":
        name = f"{args.analysis}_v{args.version}"
        out = Path(args.out) if args.out else Path(args.runs).parent / "db" / f"{name}.parquet"
        n = ingest_derived(args.runs, args.analysis, args.version, out)
        print(f"{n} derived record(s) -> {out}" if n
              else f"no derived/{name}/metrics.json under {args.runs}")
    elif args.cmd == "prune-derived":
        report = prune_derived(args.runs, args.analysis, args.version,
                               apply=not args.dry_run)
        verb = "would remove" if args.dry_run else "removed"
        print(f"{verb} {report['n_removed']} {args.analysis}_v{args.version} "
              f"dir(s) / {report['bytes_freed'] / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
