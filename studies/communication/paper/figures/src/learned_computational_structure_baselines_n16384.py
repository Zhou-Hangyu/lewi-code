"""Figure (appendix): learned_computational_structure_baselines read at n = 2^14 (the size of the authors' second main-figure
read); the same six arms, rows and columns. Output: learned_computational_structure_baselines_n16384.{pdf, svg, png} + .csv."""
from __future__ import annotations

import learned_computational_structure_baselines as B


def main() -> None:
    B.main(16384, "learned_computational_structure_baselines_n16384")


if __name__ == "__main__":
    main()
