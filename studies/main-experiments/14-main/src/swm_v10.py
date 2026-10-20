"""Lane X of the CSFRBench v1.1 final campaign (14-main): external validation on the world models of 09-swm-lewi,
every arm built through the v1.1 package under the campaign's readout rule (GOAL section 8).

13-main's lane-X runner (itself a copy of next-gen's ng_swm.py) with three changes for v1.1:

  1. the v1.1 implementation snapshot (versions/v1.1/implementation) is first on sys.path and in PYTHONPATH
     (pkgpath) and its csfrbench / runstore packages are imported before v05_port: v05_port puts the repository
     src ahead on sys.path and imports csfrbench at import time, and the pinned package wins because submodules
     resolve through the package already loaded; asserted after the import;
  2. the harness's fixed hyper-parameters (v05_port.FIXED_HP, merged into every hpo trial and every final instance
     by lewibench.train_one as hp_defaults(arm) | hp) come from the campaign config's methods block
     (configs/v1_1.yaml: readout: structural for every arm with a mechanism on the readout, p_open_init for l0hc;
     nothing for dense), so every cell runs the campaign's readout rule and every record carries the pin;
  3. 09's vector-output shim is re-patched so its Erb wrapper forwards keyword arguments: v1.1's ErbPositions.build
     passes readout_structural to SpatialMLP (the backbone wrapper already forwards **kwargs).

Roster: the nine 09 baselines dense, l1, l2, l0hc, dnw, rigl, fiedler, bimt, erb (registry keys through
v05_port.REGISTRY_KEY, as 09 uses them) and lewi-v1, registered at run time exactly as the gate
src/test_swm_readout_rule.py does. 09-swm-lewi is read-only: its harness (lewibench, v05_port, swm_tasks, analyze)
is imported, the baselines run through the same run_hpo_cell / run_final_cell, and every write lands under
14-main/external. Protocol = 09's scaled contract (results/contract_v2.json: 128x4, 128,000 Adam steps, batch 32,
the XL corpora, the 7-size grid 8,192 ... 524,288), TRIALS scrambled-Sobol trials on each arm's own frozen 09 space
(results/spaces_final_v2.json; lewi-v1 searches lewi's lr space, its only number), selected by 09's validation key,
then SEEDS paired seeds of the held-out "final" bank, so every contrast is seed-paired. Nothing is reused from the
07/09/12-extra/13-main records: the read contrasts only arms run here.

Structural parameter counts are 09's own (v05_port.structural_params: every parameter outside linear_layers()).
13-main monkeypatched a helper over lewibench.structural_params that excluded output_layer for every arm, because
its Erb readout was a plain DenseLinear that 09's count treated as structural (SpatialMLP has no linear_layers()).
Under the v1.1 rule the readout's method parameters are structural: v1.1's Erb readout is positioned (no
output_layer attribute; its weights are inverse distances between the last hidden units and the positioned output
units) and LeWi-v1's readout carries the output units' positions, so 09's definition is the right count and the
13-main helper is dropped (it would not even run on v1.1's Erb model).

Verbs::

    units                 pending cell lines (hpo first, then the finals whose hpo landed)
    hpo <task> <arm> <n>  / final <task> <arm> <n>   one cell (run inside a shard)
    sbatch [<cell line>]  one placement round; with a cell line, one job for that cell alone (the smoke), free_qos if a slot
                          exists, else paid_qos within the cap
    feed [every_s]        keep the pending cells on the queue within the lane's caps
    status / read         progress; the paired reading -> external/results/read.json + contrasts.csv

In this archive only the import-time part of the module is kept (the package pin, the fixed hyperparameters, the
vector-output shim); its scheduler verbs are not included.
"""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

import pkgpath  # noqa: F401  (the v1.1 implementation first on sys.path and in PYTHONPATH)
import csfrbench  # noqa: E402  pinned from the v1.1 implementation before v05_port puts the repository src ahead
import runstore  # noqa: E402,F401  (csfrbench.train writes its record through runstore: pinned for the same reason)

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]                                  # studies/main-experiments/14-main
S09 = HERE.parents[2] / "09-swm-lewi"
for p in (str(S09 / "src"), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
import v05_port  # noqa: E402  (repo src on sys.path)

assert str(Path(csfrbench.__file__).resolve()).startswith(pkgpath.PKG + os.sep), \
    f"csfrbench imported from {csfrbench.__file__}, not the v1.1 implementation {pkgpath.PKG}"

v05_port.REGISTRY_KEY.setdefault("lewi-v1", "lewi-v1")   # the gate's registration: the package's v1.1 method
BASELINES = ("dense", "l1", "l2", "l0hc", "dnw", "rigl", "fiedler", "bimt", "erb")   # 09's competitive roster
CANDIDATE = "lewi-v1"
ARMS = (*BASELINES, CANDIDATE)                             # unit order: the baselines first, lewi-v1 after them
assert BASELINES == tuple(v05_port.COMPETITIVE)

CAMPAIGN_CONFIG = Path(pkgpath.PKG) / "csfrbench" / "configs" / "v1_1.yaml"
CAMPAIGN_PINS = yaml.safe_load(CAMPAIGN_CONFIG.read_text())["methods"]   # registry id -> the campaign's fixed method arguments
for _arm in ARMS:                                          # reach every hpo trial and final instance through lewibench.train_one
    v05_port.FIXED_HP[_arm] = dict(CAMPAIGN_PINS[v05_port.REGISTRY_KEY[_arm]] or {})
assert v05_port.hp_defaults(CANDIDATE) == {"readout": "structural"}


@contextmanager
def _vector_output(out_dim: int):
    """09's out_dim shim (v05_port._vector_output) with the Erb wrapper forwarding keyword arguments, so
    v1.1's ErbPositions.build can pass readout_structural to SpatialMLP; the backbone wrapper is 09's."""
    saved = {}
    for name in v05_port._BACKBONES:
        cls = getattr(v05_port._nns, name)
        saved[name] = cls.__init__

        def patched(self, *args, __orig=cls.__init__, **kwargs):
            kwargs.setdefault("out_dim", out_dim)
            return __orig(self, *args, **kwargs)

        cls.__init__ = patched
    erb_saved = v05_port._erb.SpatialMLP.__init__

    def erb_patched(self, layer_sizes, generator, __orig=erb_saved, **kw):
        sizes = list(layer_sizes)
        sizes[-1] = out_dim
        return __orig(self, sizes, generator, **kw)

    v05_port._erb.SpatialMLP.__init__ = erb_patched
    try:
        yield
    finally:
        for name in v05_port._BACKBONES:
            getattr(v05_port._nns, name).__init__ = saved[name]
        v05_port._erb.SpatialMLP.__init__ = erb_saved


v05_port._vector_output = _vector_output                  # build_model resolves the name at call time (lewibench imports build_model by name)
