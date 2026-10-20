#!/usr/bin/env python3
"""Model selection and statistical aggregation from the raw per-trial records, written out as tables.

    python scripts/select_and_aggregate.py [--out DIR]

Applies the paper's selection rule to the per-trial unit tables (studies/main-experiments/17-main-clean/results/units_*.parquet
and the input rule's re-run tables) and writes, under --out (default results/aggregates/):
  winners_d8_recomputed.csv          one row per (setting, arm, instance, size): the validation-selected trial and its test NMSE
  aulc_instances_d8_recomputed.csv   AULC per (setting, arm, instance) over the setting's size window
The same functions run inside scripts/verify_results.py, which compares these tables with the shipped ledger
(studies/main-experiments/18-main-d8/ledger/d8/) and the aggregate tables (analysis/d8/contract/). The campaign's own
implementation of the rule is studies/main-experiments/17-main-clean/src/m17_analysis.py (winners) and
studies/main-experiments/18-main-d8/src/port.py (the ledgers and contract tables); this script is the short, readable form.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_results as V  # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=V.ROOT / "results/aggregates")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    winners = V.section_A(quick=False)
    winners.to_csv(a.out / "winners_d8_recomputed.csv", index=False)
    ia = V.section_B("d = 8", V.S18 / "ledger/d8/winners.csv", V.S18 / "analysis/d8/contract", claims=False)
    ia.to_csv(a.out / "aulc_instances_d8_recomputed.csv", index=False)
    print(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
