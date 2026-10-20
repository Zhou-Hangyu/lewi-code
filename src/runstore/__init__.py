"""Run shards and the results table.

One finished run = one ``run.json`` shard written into its run directory by the
executor (the shard is the source of truth: spec + metrics + validation curve +
provenance). ``python -m runstore ingest`` folds every shard under a runs root
into a single parquet table, rebuilt from scratch on each call; ``status``
counts done/failed/pending cells straight from the run directories.

Between waves, ``doctor`` validates every shard and quarantines corrupt ones
(rename to ``run.json.corrupt``) so the dispatch rule — pending means
``spec.json`` without ``run.json`` — sees those cells as pending again, and
sweeps temporary files leaked by interrupted atomic writes. After a successful
ingest, ``prune`` removes the per-run files whose content the shard already
carries, leaving 2 inodes per unretained run (directory + ``run.json``).

Query the table with DuckDB or pandas:

    duckdb -c "SELECT method, count(*) FROM 'results/<c>/db/runs.parquet' GROUP BY 1"
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

SHARD_SCHEMA_VERSION = 2     # v2: val_curve lives in the shard; results.json/events.jsonl retired
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

_REPO = Path(__file__).resolve().parents[2]      # src/runstore -> repo root


def now_iso() -> str:
    """UTC timestamp for shard fields."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def capture_provenance() -> dict[str, Any]:
    """Best-effort host, library, and git identity for one shard."""
    provenance: dict[str, Any] = {
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "torch_version": None,
        "git_commit": None,
        "git_dirty": None,
    }
    try:
        import torch

        provenance["torch_version"] = torch.__version__
    except Exception:  # noqa: BLE001 -- provenance must never fail a run
        pass
    try:
        head = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10, check=False)
        if head.returncode == 0:
            provenance["git_commit"] = head.stdout.strip()
            porcelain = subprocess.run(["git", "-C", str(_REPO), "status", "--porcelain"],
                                       capture_output=True, text=True, timeout=10, check=False)
            if porcelain.returncode == 0:
                provenance["git_dirty"] = bool(porcelain.stdout.strip())
    except Exception:  # noqa: BLE001
        pass
    return provenance


@dataclass
class RunRecord:
    """One run's shard, exactly as persisted to ``run.json``."""

    run_id: str
    run_label: str
    schema_version: int
    method_id: str
    task_id: str
    namespace: str
    n_train: int
    seed: int
    task_seed: int
    sweep_id: str | None
    spec: dict
    metrics: dict
    status: str
    finished_at: str
    source: str
    run_dir: str
    artifacts: list
    val_curve: list = field(default_factory=list)    # [[step, val_nmse, train_nmse], ...]
    hostname: str | None = None
    python_version: str | None = None
    torch_version: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None

    def write_shard(self, run_dir: str | Path) -> Path:
        path = Path(run_dir) / "run.json"
        atomic_write_text(path, json.dumps(dataclasses.asdict(self), indent=2,
                                           sort_keys=True, default=str) + "\n")
        return path


def atomic_write_text(path: Path, text: str) -> None:
    """Write text via a same-directory temporary file, fsync, and atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


# ------------------------------------------------------------------ census --

# Stems whose interrupted atomic writes can leak `.<stem>.<random>` temporaries.
_TEMP_STEMS = ("run.json", "spec.json", "results.json", "events.jsonl",
               "init.pt", "best.pt", "val_errors.npz")


def _is_leaked_temp(name: str) -> bool:
    return any(name.startswith(f".{stem}.") for stem in _TEMP_STEMS)


def _census(runs_root: str | Path) -> dict[Path, set[str]]:
    """One filesystem walk: every directory holding run-cell files, with its
    file names. A cell is visible through ``spec.json``, ``run.json``, a
    quarantined ``*.corrupt`` twin, or a leaked temporary. The versioned
    derived-analysis layer (``derived/`` inside a run directory) is never a
    cell: whatever an analysis writes there, doctor and prune leave alone."""
    cells: dict[Path, set[str]] = {}
    for dirpath, dirnames, filenames in os.walk(runs_root):
        if "derived" in dirnames:
            dirnames.remove("derived")
        interesting = ("spec.json" in filenames or "run.json" in filenames
                       or any(name.endswith(".corrupt") or _is_leaked_temp(name)
                              for name in filenames))
        if interesting:
            cells[Path(dirpath)] = set(filenames)
    return cells


# ---------------------------------------------------------------- validate --

_REQUIRED_SHARD_KEYS = ("run_id", "schema_version", "status", "spec",
                        "metrics", "finished_at")


def validate_shard(shard: Any, run_dir: str | Path | None = None) -> list[str]:
    """Problems that make a shard unusable as the run's record (empty = valid).

    With ``run_dir``, declared checkpoint artifacts (`format: pt`) must exist
    there and be nonempty — a completed run whose checkpoint is missing must
    not count as done. Prunable artifacts (metrics/events figures) are not
    checked; ``prune`` removes them by design.
    """
    if not isinstance(shard, dict):
        return ["shard is not a JSON object"]
    problems = [f"missing key {key!r}" for key in _REQUIRED_SHARD_KEYS if key not in shard]
    if problems:
        return problems
    if shard["status"] not in (STATUS_COMPLETED, STATUS_FAILED):
        problems.append(f"unknown status {shard['status']!r}")
    if not isinstance(shard["spec"], dict):
        problems.append("spec is not a mapping")
    if not isinstance(shard["metrics"], dict):
        problems.append("metrics is not a mapping")
    if run_dir is not None:
        for artifact in shard.get("artifacts") or ():
            if not isinstance(artifact, dict) or artifact.get("format") != "pt":
                continue
            name = Path(str(artifact.get("uri", "")).removeprefix("file://")).name
            path = Path(run_dir) / name
            if not path.is_file() or path.stat().st_size == 0:
                problems.append(f"declared checkpoint {name} missing or empty")
    return problems


# ------------------------------------------------------------------ doctor --

def _quarantine(path: Path) -> Path:
    """Rename a defective record out of the way without clobbering earlier evidence."""
    target = path.with_name(path.name + ".corrupt")
    index = 1
    while target.exists():
        target = path.with_name(f"{path.name}.corrupt.{index}")
        index += 1
    path.rename(target)
    return target


def doctor(runs_root: str | Path, *, min_age_s: float = 3600.0,
           apply: bool = True) -> dict[str, Any]:
    """Shard-integrity sweep: quarantine defective records, remove leaked temps.

    A quarantined ``run.json`` makes its cell pending again under the dispatch
    rule (``spec.json`` present, ``run.json`` absent), so the next materialize +
    dispatch re-runs it. ``min_age_s`` guards the temp sweep against racing a
    live writer; pass 0 only when no workers are running.
    """
    report: dict[str, Any] = {"n_cells": 0, "n_ok": 0, "quarantined": [],
                              "temps_removed": 0, "apply": bool(apply)}
    now = time.time()
    for run_dir, names in sorted(_census(runs_root).items()):
        report["n_cells"] += int("spec.json" in names or "run.json" in names)
        defects: list[tuple[Path, str]] = []
        if "run.json" in names:
            shard_path = run_dir / "run.json"
            try:
                shard = json.loads(shard_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                defects.append((shard_path, f"unreadable JSON ({exc})"))
            else:
                problems = validate_shard(shard, run_dir)
                if problems:
                    defects.append((shard_path, "; ".join(problems)))
        if "spec.json" in names:
            spec_path = run_dir / "spec.json"
            try:
                json.loads(spec_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                defects.append((spec_path, f"unreadable JSON ({exc})"))
        for path, reason in defects:
            target = _quarantine(path) if apply else path.with_name(path.name + ".corrupt")
            report["quarantined"].append({"path": str(path), "reason": reason,
                                          "moved_to": str(target)})
        if not defects and ("run.json" in names or "spec.json" in names):
            report["n_ok"] += 1
        for name in names:
            if _is_leaked_temp(name):
                temp = run_dir / name
                try:
                    if now - temp.stat().st_mtime >= float(min_age_s):
                        if apply:
                            temp.unlink()
                        report["temps_removed"] += 1
                except OSError:
                    pass
    return report


# ------------------------------------------------------------------- prune --

_PRUNE_ALWAYS = ("run.log", "results.json", "events.jsonl")


def prune(runs_root: str | Path, *, apply: bool = True) -> dict[str, Any]:
    """Drop per-run files whose content the (valid) shard already carries.

    Every pruned run keeps ``run.json``. Runs that declare checkpoint
    artifacts additionally keep ``spec.json`` — the checkpoint consumers
    (probe drivers, model rebuilds, connectivity re-renders) read it — so a
    checkpointed directory stays a self-contained model package. The delete
    list is fixed; post-hoc files (``probes*.json``, figures) are never
    touched. Run ``doctor`` and ``ingest`` first: invalid shards are skipped.
    """
    report: dict[str, Any] = {"n_dirs": 0, "n_pruned": 0, "n_files_removed": 0,
                              "bytes_freed": 0, "n_skipped_invalid": 0,
                              "apply": bool(apply)}
    for run_dir, names in sorted(_census(runs_root).items()):
        if "run.json" not in names:
            continue
        report["n_dirs"] += 1
        try:
            shard = json.loads((run_dir / "run.json").read_text())
        except (OSError, json.JSONDecodeError):
            report["n_skipped_invalid"] += 1
            continue
        if validate_shard(shard, run_dir):
            report["n_skipped_invalid"] += 1
            continue
        checkpointed = any(isinstance(a, dict) and a.get("format") == "pt"
                           for a in shard.get("artifacts") or ())
        doomed = [name for name in _PRUNE_ALWAYS if name in names]
        if not checkpointed and "spec.json" in names:
            doomed.append("spec.json")
        if not doomed:
            continue
        report["n_pruned"] += 1
        for name in doomed:
            path = run_dir / name
            try:
                report["bytes_freed"] += path.stat().st_size
                if apply:
                    path.unlink()
                report["n_files_removed"] += 1
            except OSError:
                pass
    return report


# --------------------------------------------------------------- derived ---
# The three-layer convention: (1) the immutable experiment record (run.json,
# runs.parquet); (2) retained model state (spec.json, init.pt, best.pt);
# (3) versioned derived analyses under derived/<analysis>_v<version>/, keyed
# by run_id. Training outputs are never modified by an analysis; a new
# analysis version is a new directory, never an overwrite.

def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def derived_dir(run_dir: str | Path, analysis: str, version: int) -> Path:
    if not analysis.isidentifier():
        raise ValueError(f"analysis name {analysis!r} must be a plain identifier")
    return Path(run_dir) / "derived" / f"{analysis}_v{int(version)}"


def write_derived(run_dir: str | Path, analysis: str, version: int, *,
                  parameters: dict, metrics: dict,
                  checkpoint: str | None = "best.pt") -> Path:
    """Record one versioned derived analysis of one run.

    Writes ``derived/<analysis>_v<version>/metrics.json`` with the manifest
    fields that make the result auditable later: the run's ``run_id``, the
    SHA-256 of the exact checkpoint analyzed, and the analysis code's git
    commit. The version directory is immutable — re-running an analysis means
    bumping its version, never overwriting an existing one. Extra artifacts
    (figures, arrays) belong beside ``metrics.json`` in the same directory.
    """
    run_dir = Path(run_dir)
    try:
        shard = json.loads((run_dir / "run.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{run_dir}: shard invalid, refusing to attach an "
                         f"analysis: unreadable run.json ({exc})") from exc
    problems = validate_shard(shard, run_dir)
    if problems:
        raise ValueError(f"{run_dir}: shard invalid, refusing to attach an "
                         f"analysis: {'; '.join(problems)}")
    target = derived_dir(run_dir, analysis, version)
    manifest_path = target / "metrics.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"{manifest_path} exists; derived analyses are versioned — bump the "
            f"version instead of overwriting {analysis}_v{version}")
    manifest = {
        "analysis": analysis,
        "version": int(version),
        "run_id": shard["run_id"],
        "checkpoint": checkpoint,
        "checkpoint_sha256": (_sha256(run_dir / checkpoint) if checkpoint else None),
        "analysis_git_commit": capture_provenance()["git_commit"],
        "created_at": now_iso(),
        "parameters": dict(parameters),
        "metrics": dict(metrics),
    }
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return target


def ingest_derived(runs_root: str | Path, analysis: str, version: int,
                   out_path: str | Path | None = None) -> int:
    """Consolidate one derived analysis into its own table,
    ``db/<analysis>_v<version>.parquet`` — joined to ``runs.parquet`` by
    ``run_id``. ``runs.parquet`` itself is never widened by evolving
    diagnostics; promotion into the main schema is a deliberate, separate act.
    """
    runs_root = Path(runs_root)
    name = f"{analysis}_v{int(version)}"
    rows = []
    for manifest_path in sorted(runs_root.rglob(f"derived/{name}/metrics.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skipping unreadable derived manifest {manifest_path}: {exc}")
            continue
        row = {"run_id": manifest.get("run_id"),
               "checkpoint": manifest.get("checkpoint"),
               "checkpoint_sha256": manifest.get("checkpoint_sha256"),
               "analysis_git_commit": manifest.get("analysis_git_commit"),
               "created_at": manifest.get("created_at"),
               "derived_dir": str(manifest_path.parent)}
        for key, value in (manifest.get("parameters") or {}).items():
            row[f"param_{key}"] = value
        for key, value in (manifest.get("metrics") or {}).items():
            if value is None or isinstance(value, (bool, int, float, str)):
                row[f"{key}"] = value
        rows.append(row)
    if not rows:
        return 0
    import duckdb
    import pandas as pd

    out = Path(out_path) if out_path else runs_root.parent / "db" / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    connection = duckdb.connect()
    connection.register("derived", frame)
    connection.execute(f"COPY derived TO '{out.as_posix()}' (FORMAT PARQUET)")
    connection.close()
    return len(rows)


def prune_derived(runs_root: str | Path, analysis: str, version: int,
                  *, apply: bool = True) -> dict[str, Any]:
    """Retire one derived analysis version: remove every
    ``derived/<analysis>_v<version>/`` directory under the runs root.

    Retirement is safe by construction — the immutable record and the retained
    model state never reference derived artifacts, and the analysis's own
    parquet (``db/<analysis>_v<version>.parquet``) can simply be deleted with
    it or rebuilt from the surviving versions."""
    import shutil

    name = f"{analysis}_v{int(version)}"
    report: dict[str, Any] = {"n_removed": 0, "bytes_freed": 0, "apply": bool(apply)}
    for manifest_path in sorted(Path(runs_root).rglob(f"derived/{name}/metrics.json")):
        target = manifest_path.parent
        report["bytes_freed"] += sum(p.stat().st_size for p in target.rglob("*")
                                     if p.is_file())
        if apply:
            shutil.rmtree(target)
            parent = target.parent            # drop an emptied derived/ folder
            if parent.name == "derived" and not any(parent.iterdir()):
                parent.rmdir()
        report["n_removed"] += 1
    return report


# ------------------------------------------------------------------ ingest --

_SPEC_KEYS = ("d", "n", "function_seed", "dataset_seed", "task_preset",
              "hpo_trial", "final_instance", "test_evaluation")


def iter_shards(runs_root: str | Path) -> Iterator[tuple[Path, dict]]:
    """Yield every parseable ``run.json`` under ``runs_root``; report the rest.
    Files inside the derived-analysis layer are never shards."""
    for path in sorted(Path(runs_root).rglob("run.json")):
        if "derived" in path.parts:
            continue
        try:
            yield path, json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skipping unreadable shard {path}: {exc}")


def flatten(shard: dict, path: Path | None = None) -> dict[str, Any]:
    """One shard -> one flat table row (scalars, plus the validation curve as
    parallel list columns; the shard keeps the rest).

    ``run_dir`` is the shard's CURRENT location (from ``path``) so the table
    survives a results-tree move; the shard itself keeps the historical path.
    """
    row: dict[str, Any] = {
        "run_id": shard.get("run_id"),
        "run_label": shard.get("run_label"),
        "task": shard.get("task_id"),
        "method": shard.get("method_id"),
        "status": shard.get("status"),
        "sweep_id": shard.get("sweep_id"),
        "seed": shard.get("seed"),
        "task_seed": shard.get("task_seed"),
        "finished_at": shard.get("finished_at"),
        "git_commit": shard.get("git_commit"),
        "git_dirty": shard.get("git_dirty"),
        "run_dir": str(path.parent) if path is not None else shard.get("run_dir"),
    }
    spec = shard.get("spec") or {}
    for key in _SPEC_KEYS:
        if key in spec:
            row[key] = spec[key]
    for key, value in (spec.get("hp") or {}).items():
        row[f"hp_{key}"] = value
    for key, value in (shard.get("metrics") or {}).items():
        if value is None or isinstance(value, (bool, int, float, str)):
            row[key] = value
    curve = shard.get("val_curve") or []
    if curve:
        row["val_curve_step"] = [int(point[0]) for point in curve]
        row["val_curve_val_nmse"] = [None if point[1] is None else float(point[1])
                                     for point in curve]
        row["val_curve_train_nmse"] = [None if point[2] is None else float(point[2])
                                       for point in curve]
    return row


def ingest(runs_root: str | Path, out_path: str | Path) -> int:
    """Rebuild the parquet table from every valid shard under ``runs_root``."""
    rows = []
    skipped = 0
    for path, shard in iter_shards(runs_root):
        problems = validate_shard(shard)
        if problems:
            skipped += 1
            print(f"skipping invalid shard {path}: {'; '.join(problems)}")
            continue
        rows.append(flatten(shard, path))
    if skipped:
        print(f"{skipped} invalid shard(s) skipped; run `python -m runstore doctor`")
    if not rows:
        return 0
    import duckdb
    import pandas as pd

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    connection = duckdb.connect()
    connection.register("runs", frame)
    connection.execute(f"COPY runs TO '{out.as_posix()}' (FORMAT PARQUET)")
    connection.close()
    return len(rows)


def status(runs_root: str | Path) -> str:
    """Done/failed/pending counts per (group, task, method) from the directories.

    A pruned run keeps only ``run.json``, so the census is the union of
    directories holding ``spec.json`` or ``run.json``.
    """
    runs_root = Path(runs_root)
    counts: dict[tuple[str, str, str], dict[str, int]] = {}
    for run_dir, names in _census(runs_root).items():
        if "spec.json" not in names and "run.json" not in names:
            continue
        relative = run_dir.relative_to(runs_root)
        group = relative.parts[0] if len(relative.parts) > 1 else "."
        task = method = "?"
        shard = None
        if "run.json" in names:
            try:
                shard = json.loads((run_dir / "run.json").read_text())
            except (OSError, json.JSONDecodeError):
                shard = None
        try:
            spec = (json.loads((run_dir / "spec.json").read_text())
                    if "spec.json" in names else (shard or {}).get("spec", {}))
            task, method = str(spec.get("task")), str(spec.get("method"))
        except (OSError, json.JSONDecodeError):
            pass
        if "run.json" in names:
            state = str(shard.get("status", "?")) if isinstance(shard, dict) else "corrupt"
        else:
            state = "pending"
        bucket = counts.setdefault((group, task, method), {})
        bucket[state] = bucket.get(state, 0) + 1

    if not counts:
        return f"no run cells under {runs_root}"
    states = ("completed", "failed", "pending")
    lines = []
    totals = {state: 0 for state in states}
    for (group, task, method), bucket in sorted(counts.items()):
        extras = {state: n for state, n in bucket.items() if state not in states}
        cell = "  ".join(f"{state}={bucket.get(state, 0)}" for state in states)
        if extras:
            cell += "  " + "  ".join(f"{state}={n}" for state, n in sorted(extras.items()))
        lines.append(f"{group:>8}  {task:<20} {method:<8} {cell}")
        for state in states:
            totals[state] += bucket.get(state, 0)
    lines.append("total     " + "  ".join(f"{state}={totals[state]}" for state in states))
    return "\n".join(lines)
