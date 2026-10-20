"""Rebuild every paper figure from src/ and data/ into out/.

    python build.py            rebuild all
    python build.py templates  rebuild the figures whose stem contains the word

Wave 2 (13-main, CSFRBench v1.0) builds by name into out-v2/ from data-v2/
with the three switches of src/figstyle.py and src/paperstyle.py:

    CSFR_FIGURES_OUT=out-v2 CSFR_FIGURES_DATA=data-v2 CSFR_FIGURES_LEWI=lewi-v1 \
        python build.py <stem> ...

out-v2 without data-v2 (or data-v2 without out-v2) is refused: the 08-main
data would render into the wave-2 folder, or the wave-2 data over the 08-main
originals in out/.

Wave 3 (14-main, CSFRBench v1.1) builds the same way into out-main-14/ from
data-main-14/ (decision 2026-09-12: every paper-facing figure of 14-main lives in
out-main-14/):

    CSFR_FIGURES_OUT=out-main-14 CSFR_FIGURES_DATA=data-main-14 CSFR_FIGURES_LEWI=lewi-v1 \
        python build.py <stem> ...

Every data folder renders into its own output folder (figstyle.OUT_OF); any
other pairing is refused.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from figstyle import DATA, OUT, OUT_OF  # noqa: E402

BUILDERS = (
    # the manuscript's experimental figures and tables (CSFR_FIGURES_DATA=data-paper-d8 CSFR_FIGURES_OUT=out-paper/d8
    # CSFR_FIGURES_LEWI=lewi-v1-rgm; data-paper-d32 / out-paper/d32 for the nuisance-input curves); scripts/rebuild_figures.sh
    "graph_templates", "random_graphs", "graph_oracle_masks", "local_fit",
    "sample_efficiency", "sample_efficiency_aulc", "learning_curves_all",
    "learned_computational_structure_n16384", "learned_computational_structure_baselines_n16384", "representation_diagnostics_across_n",
    "fig6", "fig6_sensitivity", "structure_modules", "structure_alignment_vs_n", "connection_density_table", "connection_density",
    "embeddings_stack", "world_model_grid_compact", "world_model_grid", "world_model_transfer_all",
    "aulc_table", "additional_structures_table", "wiring_alignment_table", "world_model_hpo_table")


def main(argv: list[str]) -> None:
    if OUT_OF.get(DATA.name) != OUT.relative_to(DATA.parent).as_posix():
        raise SystemExit(f"[build] CSFR_FIGURES_DATA={DATA.name} renders into {OUT_OF.get(DATA.name)}, not "
                         f"CSFR_FIGURES_OUT={OUT.name}: a wave's data never mix with another wave's folder "
                         f"(pairs: {OUT_OF})")
    wanted = [b for b in BUILDERS
              if not argv or any(a in b for a in argv)]
    if not wanted:
        raise SystemExit(f"no figure matches {argv}; known: {list(BUILDERS)}")
    for name in wanted:
        try:
            __import__(name).main()
        except FileNotFoundError as exc:
            missing = Path(str(exc.filename or ""))
            if DATA.name != "data" and str(DATA) in str(missing):
                raise SystemExit(f"[build] {name}: no wave-2 data yet - {missing.relative_to(DATA.parent)} missing "
                                 "(the harvesters of README, Wave 2, run first)") from None
            raise


if __name__ == "__main__":
    main(sys.argv[1:])
