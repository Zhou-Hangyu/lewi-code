"""Figure: learned_computational_structure measured at n = 2^14 (authors' request, 2026-09-18).

The same two blocks as learned_computational_structure (linear probes on the depth-4 tree, left; joint causal
interventions, right; Dense, LeWi and GraphOracle(all); means over 16 instances with 95% Student-t intervals),
read at training-set size n = 16,384 instead of 2^15. No new fit was run: the twin's linear probes cover every
winner at every size and its joint causal fits on the tree cover the three arms at every size.

Decision 2026-09-22: the compact layout - the heatmap row alone at the two-row figure's heatmap size, the two blocks
0.28 in apart, no level legend (the depth curves of this size are the stem learned_computational_structure_n16384_merged).

Data: as learned_computational_structure (data/learned_structure/probe_values.csv, data/causal_nopatch/cells.csv).
Output: learned_computational_structure_n16384.{pdf, svg, png} and its aggregates .csv.
"""
from __future__ import annotations

import learned_computational_structure as L


def main() -> None:
    L.main(16384, "learned_computational_structure_n16384", curves=False)


if __name__ == "__main__":
    main()
