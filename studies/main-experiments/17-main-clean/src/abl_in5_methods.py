"""The mechanism ablations and the initialization-scale arms of the final LeWi on the adopted arm's initialization
(18-main-d8, decision 2026-09-22: the paper's ablation figures in the same condition as every other LeWi result).

Every arm here is its abl_methods twin built from ``lewi-v1-rgm-in5`` (18-main-d8/input-cutoff/src/cutoff_methods.py: an
input with fewer than five open first-transition connections at initialization has its position drawn again; the readout's
own rejection rule, gate and mask are the arm's) instead of the record's ``lewi-v1-rgm``: the input rule fires in the parent
build, the variant's change follows, exactly as in abl_methods. The sensitivity arms of the position dimension, the radius
and the seed replicates need no class of their own: their panels map the LeWi key to ``lewi-v1-rgm-in5`` (m17.PANELS
arm_method) and keep their fixed hyperparameters.

  ``lewi-abl-soft-in5``  ``lewi-abl-fixed-in5``  ``lewi-abl-random-in5``  ``lewi-abl-ste-in5``
  ``lewi-init05-in5``  ``lewi-init2-in5``
"""
from __future__ import annotations

import sys
from pathlib import Path

from csfrbench.registry import method

import abl_methods

CUTOFF_SRC = Path(__file__).resolve().parents[2] / "18-main-d8" / "input-cutoff" / "src"
if str(CUTOFF_SRC) not in sys.path:
    sys.path.append(str(CUTOFF_SRC))
import cutoff_methods  # noqa: E402


class _In5(cutoff_methods._InputCondition):
    min_open = 5


BASE = {"lewi-abl-soft": abl_methods.LeWiSoftOnly, "lewi-abl-fixed": abl_methods.LeWiFixedEmbeddings,
        "lewi-abl-random": abl_methods.LeWiRandomSparse, "lewi-abl-ste": abl_methods.LeWiStraightThrough,
        "lewi-init05": abl_methods.LeWiInitHalf, "lewi-init2": abl_methods.LeWiInitDouble}
IN5 = {arm: f"{arm}-in5" for arm in BASE}
ARMS = tuple(IN5.values())
for _arm, _cls in BASE.items():
    # method resolution: the variant's build, then the input rule's, then the record's
    method(IN5[_arm])(type(f"{_cls.__name__}In5", (_cls, _In5), {}))
