"""Quantitative claims of the submitted draft (draft.pdf, 2026-09-24, 35 pages), transcribed from its text.

Every entry: an id, where it stands in the draft (page, margin line numbers), the claim, and the record that must
reproduce it in the package. The recompute script of the package (scripts/verify_results.py) and the
paper-to-code audit both read this list. Numbers copied from tables are in the table entries; figure values are
checked from the figure data files.
"""

CLAIMS = [
    # ---------------------------------------------------------------- abstract / introduction
    dict(id="abs_tasks", page=1, lines="025-027", claim="17 world-model prediction tasks; LeWi improves over Dense on 16; wins 99 of 119 comparisons against seven principal baselines",
         values=dict(tasks=17, dense_wins=16, wins=99, pairs=119), record="studies/communication/paper/figures/data-paper-d8/world_model_transfer_all/paired.csv"),
    dict(id="intro_reductions", page=2, lines="083-085", claim="Relative to Dense, LeWi reduces mean AULC by 20% on the depth-4 tree and 34% on the reuse DAG, and by 27% across 32 independently sampled depth-4 DAGs",
         values=dict(tree=0.20, dag=0.34, rand=0.27, rand_instances=32), record="studies/main-experiments/18-main-d8/analysis/d8/contract/aulc.csv"),
    dict(id="intro_f1_ratio", page=2, lines="089-090", claim="more than three times the edge F1 of initialization and randomized controls across four structured settings",
         values=dict(ratio_min=3.0, settings=4), record="studies/main-experiments/18-main-d8/structure/tables/summary_2p15.csv, wiring_alignment.csv"),
    dict(id="intro_wm", page=2, lines="092-093", claim="across 17 world-model tasks, LeWi wins 16 of 17 comparisons against Dense and 99 of 119 against seven principal baselines",
         values=dict(dense_wins=16, tasks=17, wins=99, pairs=119), record="studies/communication/paper/figures/data-paper-d8/world_model_transfer_all/paired.csv"),
    # ---------------------------------------------------------------- section 6.1 setup
    dict(id="setup_backbone", page=6, lines="291-293", claim="width 32 with four hidden layers, except the depth-5 tree: width 64 with eight hidden layers",
         values=dict(width=32, hidden=4, width5=64, hidden5=8), record="studies/main-experiments/17-main-clean/src/m17.py (profiles)"),
    dict(id="setup_grid", page=6, lines="300-303", claim="d = 8 active inputs, arity 2, sigma^2 = 0.01; n in 2^9..2^15; 64,000 steps; depth-5 tree d = 16, n in 2^11..2^17, 128,000 steps; nuisance settings d = 32 (24 or 16 inert)",
         values=dict(d=8, k=2, sigma_sq=0.01, n_min=512, n_max=32768, steps=64000, d5=16, n5_min=2048, n5_max=131072, steps5=128000, d_nuisance=32),
         record="studies/main-experiments/17-main-clean/src/m17.py (cells, windows, profiles)"),
    dict(id="setup_trials", page=6, lines="305-308", claim="32 scrambled-Sobol trials per instance and size; validation 4,096 and test 65,536 examples",
         values=dict(T=32, n_val=4096, n_test=65536), record="studies/main-experiments/17-main-clean/results/units_*.parquet (32 trials per cell)"),
    dict(id="setup_instances", page=6, lines="312-313", claim="16 target instances in each fixed-graph setting and 32 in the sampled depth-4 DAG setting",
         values=dict(fixed=16, sampled=32), record="studies/main-experiments/18-main-d8/ledger/d8/winners.csv"),
    # ---------------------------------------------------------------- section 6.2
    dict(id="s62_reductions", page=6, lines="321-324", claim="20% (depth-4 tree), 34% (reuse DAG), 27% (32 sampled depth-4 DAGs); depth-5 tree 31% lower mean AULC than Dense and better than every learned baseline on all 16 instances",
         values=dict(tree=0.20, dag=0.34, rand=0.27, tree5=0.31, tree5_wins=16), record="studies/main-experiments/18-main-d8/analysis/d8/contract/{aulc,contrasts}.csv"),
    dict(id="s62_gap", page=7, lines="361-362", claim="On the depth-4 tree LeWi closes 51% of the AULC gap between Dense and GraphOracle(all), compared with 26% for L0",
         values=dict(lewi_share=0.51, l0_share=0.26), record="studies/main-experiments/18-main-d8/analysis/d8/contract/go_gap.csv"),
    # ---------------------------------------------------------------- section 6.3 (Figure 5, n = 2^14, depth-4 tree)
    dict(id="s63_probe_gaps", page=7, lines="373-375", claim="LeWi closes roughly three quarters of the Dense-to-GraphOracle(all) probe-R^2 gap for level-2 intermediates in hidden layer 3 and about two thirds for the level-3 intermediate in layer 4",
         values=dict(level2_layer3_share=0.75, level3_layer4_share=0.67, tolerance=0.08), record="studies/communication/paper/figures/data-paper-d8 -> out-paper/d8/learned_computational_structure_n16384.csv"),
    dict(id="s63_causal_gaps", page=8, lines="401-403", claim="counterfactual R^2: level-1 intermediates in layer 1, LeWi closes nearly all of the Dense-to-GraphOracle(all) gap; level-2 in layer 3 reach approximately GraphOracle(all)",
         values=dict(level1_layer1_share_min=0.85, level2_layer3_share_min=0.9), record="studies/communication/paper/figures/data-paper-d8 -> out-paper/d8/learned_computational_structure_n16384.csv"),
    # ---------------------------------------------------------------- section 6.4
    dict(id="s64_nuisance", page=8, lines="411-413", claim="LeWi's advantage over Dense increases across all seven settings after adding 24 inert inputs",
         values=dict(settings=7), record="studies/main-experiments/18-main-d8/analysis/{d8,d32s}/contract/contrasts.csv"),
    dict(id="s64_f1", page=8, lines="427-431", claim="trained LeWi achieves more than three times the edge F1 of initialization and randomized controls; for edges originating at target intermediates mean F1 remains 0.471",
         values=dict(ratio_min=3.0, f1_intermediates=0.471), record="studies/main-experiments/18-main-d8/structure/tables/summary_2p15.csv, wiring_alignment.csv"),
    dict(id="s65_wm", page=9, lines="467-468", claim="119 pairs: LeWi wins 99, ties 6, loses 14; against Dense 16 of 17",
         values=dict(wins=99, ties=6, losses=14, dense_wins=16), record="studies/communication/paper/figures/data-paper-d8/world_model_transfer_all/paired.csv"),
    dict(id="disc_cost", page=9, lines="484", claim="training-step time 2.08x-2.40x (the optimized implementation adopted 2026-09-25; the draft of 2026-09-24 printed 2.45x-2.86x for the implementation as run in the campaigns)", values=dict(ratio_min=2.08, ratio_max=2.40), record="studies/communication/paper/figures/out-paper/d8/overhead_compact_table.csv"),
    # ---------------------------------------------------------------- appendix B
    dict(id="B1_constants", page=14, lines="722-723", claim="r = 3, sigma_e = 0.5, R^2 = 2 r sigma_e^2 = 1.5; same radius on every affine layer incl. the readout",
         values=dict(r=3, sigma_e=0.5, R2=1.5), record="src/csfrbench/methods/lewi_v1.py, cmask.py CODE_INIT"),
    dict(id="B1_init_rules", page=14, lines="725-727", claim="two rejection checks at initialization: a transition with no active connection redraws its receivers; an input with fewer than five active outgoing connections is redrawn until it has at least five",
         values=dict(min_open_input=5), record="studies/main-experiments/12-extra/src/w5_methods.py, studies/main-experiments/18-main-d8/input-cutoff/src/cutoff_methods.py"),
    # ---------------------------------------------------------------- appendix C
    dict(id="C2_topologies", page=16, lines="823-825", claim="across 4,096 development draws the sampler produces 354 distinct canonical topologies; a typical 32-instance cohort about 30",
         values=dict(draws=4096, distinct=354, per_cohort_about=30), record="src/csfrbench tasks (random graph sampler); scripts/verify_results.py recomputes"),
    dict(id="C3_localfns", page=16, lines="841-862", claim="rho = 0.5; f ~ U[0.25, 0.4]; Fourier Q = 8, sigma_omega = 0.85 (arity 2); Q = 24, sigma_omega = 1.5 (arity 8); amplitudes U[0.7, 1]",
         values=dict(rho=0.5, f_lo=0.25, f_hi=0.4, Q2=8, sw2=0.85, Q8=24, sw8=1.5, a_lo=0.7, a_hi=1.0), record="src/csfrbench/tasks"),
    dict(id="C4_pool", page=17, lines="873-880", claim="normalization pool of 2^18 inputs; a local function is redrawn only if its raw std on the pool is below 1e-3",
         values=dict(pool=262144, std_floor=1e-3), record="src/csfrbench/tasks"),
    dict(id="C6_localfit", page=17, lines="906-916", claim="local-function fits: 16 dev draws; 1-4 hidden layers; widths 2..1024; Adam 32,000 steps on 65,536 examples; 8 Sobol lrs in [1e-4, 1e-1]; 16,384 val, 65,536 test; median NMSE 0.0085 (add.-mult., 1 hidden x width 8) and 0.0016 (Fourier)",
         values=dict(draws=16, steps=32000, n_train=65536, lrs=8, n_val=16384, n_test=65536, med_anova2=0.0085, med_fourier2=0.0016), record="studies/communication/paper/figures/data-paper-d8/local_fit/a4_grid.csv"),
    dict(id="C9_probe", page=19, lines="1009-1014", claim="diagnostic pool 8,192 = 4,096 / 2,048 / 2,048; features active on fewer than 10 probe-training examples excluded; lambda in 10^-4 .. 10^7 by half decades",
         values=dict(pool=8192, split=(4096, 2048, 2048), min_active=10), record="src/csfrbench/das_probe.py"),
    dict(id="C10_das", page=20, lines="1029-1032", claim="10,000 / 1,000 / 2,000 counterfactual pairs; 18 epochs; batch 64; rotation lr 0.003; boundary lr 0.0003; validation every 200 updates; three restarts; temperature d/4096",
         values=dict(pairs=(10000, 1000, 2000), epochs=18, batch=64, lr_rot=0.003, lr_bound=0.0003, val_every=200, restarts=3), record="src/csfrbench/das_core.py"),
    dict(id="C11_training", page=20, lines="1048-1052", claim="Adam beta 0.9 / 0.999, eps 1e-8, batch 32, no weight decay; cosine decay to 1%; 64,000 steps with validation every 1,000; 128,000 steps with validation every 2,000; 64 validation checkpoints",
         values=dict(beta1=0.9, beta2=0.999, eps=1e-8, batch=32, final_lr_frac=0.01, val_every=1000, val_every5=2000, checkpoints=64), record="src/csfrbench/train.py, studies/main-experiments/17-main-clean/src/m17.py profiles"),
    dict(id="C12_selection", page=20, lines="1058-1063", claim="span of selected values expanded by 25% of its log width, split equally between the two sides; T = 32; checkpoint with lowest validation NMSE, then the configuration with lowest validation NMSE",
         values=dict(pad=0.25, T=32), record="studies/main-experiments/17-main-clean/src/m17.py (freeze) + studies/main-experiments/17-main-clean/manifests"),
    dict(id="C12_audit", page=20, lines="1066-1075", claim="boundary occupancy (within 10% of a boundary): largest 0.17 at d = 8; median normalized positions 0.31-0.76; LeWi 2% (32x4) and 3% (depth-5); depth-5 nuisance setting BIMT 33% and l1 26%",
         values=dict(max_share=0.17, median_lo=0.31, median_hi=0.76, lewi_32x4=0.02, lewi_d5=0.03, bimt_d5_nuisance=0.33, l1_d5_nuisance=0.26), record="studies/main-experiments/18-main-d8/analysis/hpo_boundary/*.csv"),
    dict(id="C13_stats", page=21, lines="1152-1157", claim="two-sided 95% Student-t intervals over instances; paired per-instance AULC differences; no p-values",
         values={}, record="src/csfrbench/metrics.py, studies/main-experiments/18-main-d8/src/final.py tci"),
    dict(id="table2", page=21, lines="1092-1123", claim="Table 2: frozen search spaces, 32x4 and depth-5 backbones (see figures/expected/d8/search_space_table.md)",
         values={}, record="studies/main-experiments/17-main-clean/manifests/spaces/<set>/ -> out-paper/d8/search_space_table"),
    dict(id="table3", page=22, lines="1135-1150", claim="Table 3: mean AULC +- 95% half-width on the six fixed settings", values={}, record="out-paper/d8/aulc_table.md vs recomputation from the unit tables and ledgers"),
    dict(id="table4", page=23, lines="1201-1230", claim="Table 4: depth-5 tree and sampled depth-4 DAGs: mean AULC, paired differences, wins", values={}, record="additional_structures_table.md vs recomputation"),
    # ---------------------------------------------------------------- appendix D
    dict(id="D4_table5", page=27, lines="1441-1450", claim="Table 5 wiring alignment at n = 2^15: depth-4 tree P .677 R .579 F1 .623 | from intermediates .503 .375 .425 | assessable recall .731; reuse DAG .886 .695 .777 | .829 .493 .613 | .772; Fourier tree .518 .446 .478 | .403 .321 .354 | .508; sampled depth-4 DAGs .670 .501 .569 | .632 .415 .492 | .533",
         values=dict(tree=(.677, .579, .623, .503, .375, .425, .731), dag=(.886, .695, .777, .829, .493, .613, .772), ftree=(.518, .446, .478, .403, .321, .354, .508), rand=(.670, .501, .569, .632, .415, .492, .533)),
         record="studies/main-experiments/18-main-d8/structure/tables/wiring_alignment.csv"),
    dict(id="D5_density", page=26, lines="1395-1403", claim="initialization density 0.61 in every setting; all-active compositional settings 0.29-0.40 with the high-arity control near initialization; with nuisance inputs 0.24-0.31 and the high-arity control 0.42; depth-5 tree near 0.42 in both; input transition 0.07-0.29 with nuisance inputs; depth-5 tree falls from 0.58 at n = 2^11 to 0.38 at n = 2^17 (all-active)",
         values=dict(init=0.61, d8_lo=0.29, d8_hi=0.40, d32_lo=0.24, d32_hi=0.31, d32_dense=0.42, tree5=0.42, input_lo=0.07, input_hi=0.29, tree5_start=0.58, tree5_end=0.38),
         record="studies/main-experiments/18-main-d8/density/tables/{summary,by_size}.csv"),
    dict(id="D6_sensitivity", page=27, lines="1500-1503", claim="r in {1,2,3,4} similar, r = 8 increases AULC; moderate R^2 changes small; sigma_e sensitivity stronger on the depth-4 tree at the larger scale",
         values={}, record="studies/communication/paper/figures/data-paper-d8/fig6/sensitivity_full.csv, r_sweep_instances.csv"),
    dict(id="D7_cost", page=28, lines="1486-1497, 1509-1511", claim="Table 6 (the optimized implementation adopted 2026-09-25): 32x4 64,000 steps: step 1.471 vs 3.049 ms (2.08x), params 3,489 vs 3,900, whole budget 97 vs 211 s, inference 0.57 vs 1.28 ms, RSS 871 vs 874 MB, GPU 2.92x; 64x8 128,000 steps: step 2.206 vs 5.301 ms (2.40x), params 30,273 vs 31,860, whole budget 340 vs 722 s, inference 3.34 vs 4.56 ms, RSS 917 vs 930 MB, GPU 3.41x; the note keeps the implementation as run in the campaigns (1.647 / 4.020 ms, 2.44x; 2.553 / 7.432 ms, 2.91x). The draft of 2026-09-24 printed the campaign implementation (1.619 / 3.964 ms, 2.45x; 2.466 / 7.046 ms, 2.86x)",
         values=dict(step32=(1.471, 3.049, 2.08), params32=(3489, 3900), budget32=(97, 211), infer32=(0.57, 1.28), rss32=(871, 874), gpu32=2.92,
                     step64=(2.206, 5.301, 2.40), params64=(30273, 31860), budget64=(340, 722), infer64=(3.34, 4.56), rss64=(917, 930), gpu64=3.41, params_lo=0.05, params_hi=0.12),
         record="studies/main-experiments/18-main-d8/overhead-fast/tables/overhead_fast.csv"),
    # ---------------------------------------------------------------- appendix E
    dict(id="E1_suite", page=29, lines="1548-1551, 1564-1565", claim="registry of 21 environments; 17 satisfy the criteria at the 256x4 protocol; state dims 2-78, action dims 1-12",
         values=dict(registry=21, counted=17, obs_lo=2, obs_hi=78, act_lo=1, act_hi=12), record="studies/main-experiments/18-main-d8/swm-reconcile/{counted_set,excluded_tasks,screen_256x4}.csv"),
    dict(id="E2_protocol", page=30, lines="1597-1614", claim="80/10/10 split by episode; n in 2^13..2^19; width 256, four hidden layers; 128,000 Adam steps, batch 32, cosine to 1%; validation every 500 steps; 32 Sobol configurations; ranking by mean validation rollout NMSE over horizons 1,2,4,8,16; 16 paired final seeds",
         values=dict(split=(0.8, 0.1, 0.1), n_min=8192, n_max=524288, width=256, hidden=4, steps=128000, batch=32, val_every=500, T=32, horizons=(1, 2, 4, 8, 16), final_seeds=16),
         record="studies/main-experiments/09-swm-lewi/src/lewibench.py, studies/main-experiments/16-main-final/src/swm16.py"),
    dict(id="E2_reselection", page=30, lines="1616-1619, 1725-1727", claim="instability trigger: geometric-mean validation rollout error of the finals >= 1.5x the search trial, or one final > 10x the cell median, and mean validation log-error exceeding both neighbouring sizes by >= 0.10; then the four highest-ranked configurations on eight fresh paired seeds, re-ranked by mean log10 validation rollout NMSE; a changed selection retrained on the same 16 seeds",
         values=dict(factor=1.5, outlier=10, log_gap=0.10, K=4, M=8, final_seeds=16), record="studies/main-experiments/18-main-d8/swm-rollout/src/{screen,resel}.py"),
    dict(id="table7", page=33, lines="1729-1746", claim="Table 7 world-model search spaces: Dense lr [8.58e-5, 4.63e-2]; l1 lr [4.94e-5, 9.85e-2] lambda [1.66e-9, 1.66e3]; l2 lr [3.44e-4, 3.98e-2] lambda [5.22e-7, 9.06e1]; L0 lr [1.30e-3, 6.50e-2] lambda [2.75e-9, 1.61e2]; RigL lr [1.41e-4, 1.60e-2] density [1.58e-2, 1]; BIMT lr [3.85e-4, 1.22e-2] lambda [1.25e-8, 7.36e-3]; Fiedler lr [1.83e-4, 1.68e-2] delta [2.03e-9, 2.84e5]; positions lr [2.15e-4, 1.35e-1]; LeWi lr [9.66e-5, 2.53e-2]",
         values=dict(dense=(8.58e-5, 4.63e-2), l1=(4.94e-5, 9.85e-2, 1.66e-9, 1.66e3), l2=(3.44e-4, 3.98e-2, 5.22e-7, 9.06e1), l0=(1.30e-3, 6.50e-2, 2.75e-9, 1.61e2),
                     rigl=(1.41e-4, 1.60e-2, 1.58e-2, 1.0), bimt=(3.85e-4, 1.22e-2, 1.25e-8, 7.36e-3), fiedler=(1.83e-4, 1.68e-2, 2.03e-9, 2.84e5), positions=(2.15e-4, 1.35e-1), lewi=(9.66e-5, 2.53e-2)),
         record="studies/main-experiments/09-swm-lewi/results/spaces_final_v2.json, studies/main-experiments/18-main-d8/swm-lrx/manifests/spaces_lrx.json"),
    dict(id="E3_tallies", page=33, lines="1757-1762", claim="7 principal baselines: 99 wins, 6 ties, 14 losses of 119; with neuron positions 116 wins, 6 ties, 14 losses of 136; Dense 16-0-1, l1 15-0-2, l2 13-0-4, L0 16-1-0, RigL 14-1-2, Fiedler 13-3-1, BIMT 12-1-4, positions 17-0-0",
         values=dict(wins=99, ties=6, losses=14, pairs=119, wins8=116, pairs8=136,
                     per_arm=dict(dense=(16, 0, 1), l1=(15, 0, 2), l2=(13, 0, 4), l0=(16, 1, 0), rigl=(14, 1, 2), fiedler=(13, 3, 1), bimt=(12, 1, 4), positions=(17, 0, 0))),
         record="studies/communication/paper/figures/data-paper-d8/world_model_transfer_all/paired.csv"),
    dict(id="fig21_values", page=34, lines="1799-1808", claim="Figure 21 cell values (mean paired delta AULC per task and baseline) and the W-T-L column",
         values={}, record="studies/communication/paper/figures/data-paper-d8/world_model_transfer_all/paired.csv (mean column)"),
]


# ---------------------------------------------------------------- the draft's tables, transcribed
# Table 3 (page 22): mean AULC +- 95% half-width; columns tree8, dag8, ftree8, flat8, fdag8, dense8
_T3_COLS = ("tree8", "dag8", "ftree8", "flat8", "fdag8", "dense8")
_T3_ROWS = {
    "lewi-v1-rgm": [(0.508, 0.062), (0.179, 0.042), (0.035, 0.013), (0.008, 0.001), (0.019, 0.006), (0.085, 0.011)],
    "noreg": [(0.633, 0.059), (0.271, 0.056), (0.071, 0.023), (0.018, 0.002), (0.043, 0.009), (0.066, 0.008)],
    "l1": [(0.626, 0.061), (0.268, 0.061), (0.044, 0.020), (0.013, 0.002), (0.027, 0.007), (0.066, 0.008)],
    "l2": [(0.593, 0.061), (0.220, 0.056), (0.046, 0.019), (0.015, 0.002), (0.027, 0.005), (0.048, 0.006)],
    "l0hc": [(0.569, 0.065), (0.232, 0.060), (0.056, 0.020), (0.027, 0.004), (0.038, 0.008), (0.089, 0.010)],
    "rigl": [(0.587, 0.060), (0.234, 0.060), (0.046, 0.016), (0.017, 0.003), (0.028, 0.006), (0.077, 0.010)],
    "bimt": [(0.592, 0.061), (0.211, 0.056), (0.040, 0.015), (0.013, 0.002), (0.023, 0.005), (0.066, 0.009)],
    "fiedler": [(0.632, 0.054), (0.285, 0.058), (0.068, 0.023), (0.022, 0.003), (0.042, 0.009), (0.063, 0.007)],
    "erb_positions": [(0.861, 0.030), (0.643, 0.059), (0.213, 0.055), (0.233, 0.032), (0.204, 0.051), (0.404, 0.059)],
    "go1": [(0.491, 0.055), (0.085, 0.026), (0.040, 0.013), None, (0.015, 0.003), None],
    "graph_oracle": [(0.388, 0.054), (0.044, 0.012), (0.025, 0.008), (0.003, 0.000), (0.009, 0.001), None],
}
TABLE3 = {arm: {c: v for c, v in zip(_T3_COLS, vals) if v is not None} for arm, vals in _T3_ROWS.items()}

# Table 4 (page 23): (AULC mean, half-width[, contrast mean, lo, hi, LeWi wins]) per arm; depth-5 tree (16 instances) and sampled depth-4 DAGs (32)
TABLE4 = {
    "tree5d16": {
        "lewi-v1-rgm": (0.812, 0.037), "noreg": (1.185, 0.572, -0.374, -0.950, 0.203, 16), "l1": (0.905, 0.020, -0.093, -0.114, -0.073, 16),
        "l2": (0.895, 0.021, -0.084, -0.103, -0.065, 16), "l0hc": (0.882, 0.025, -0.070, -0.089, -0.052, 16), "rigl": (0.884, 0.023, -0.072, -0.090, -0.054, 16),
        "bimt": (0.903, 0.020, -0.091, -0.111, -0.071, 16), "fiedler": (0.913, 0.018, -0.102, -0.124, -0.080, 16), "erb_positions": (0.992, 0.003, -0.181, -0.215, -0.146, 16),
        "go1": (0.717, 0.038, 0.095, 0.079, 0.110, 0), "go2": (0.531, 0.047, 0.280, 0.255, 0.305, 0), "graph_oracle": (0.458, 0.051, 0.353, 0.326, 0.381, 0)},
    "rand8": {
        "lewi-v1-rgm": (0.197, 0.030), "noreg": (0.268, 0.033, -0.072, -0.080, -0.063, 32), "l1": (0.244, 0.031, -0.047, -0.056, -0.038, 32),
        "l2": (0.245, 0.033, -0.049, -0.059, -0.039, 32), "l0hc": (0.245, 0.033, -0.048, -0.056, -0.040, 32), "rigl": (0.241, 0.032, -0.044, -0.053, -0.035, 31),
        "bimt": (0.247, 0.033, -0.050, -0.061, -0.040, 31), "fiedler": (0.268, 0.033, -0.071, -0.081, -0.062, 32), "erb_positions": (0.503, 0.037, -0.307, -0.340, -0.273, 32),
        "go1": (0.206, 0.030, -0.009, -0.018, 0.000, 20), "go2": (0.151, 0.024, 0.046, 0.035, 0.056, 1), "graph_oracle": (0.111, 0.022, 0.086, 0.072, 0.100, 0)},
}

# Table 5 (page 27): all edges P R F1 | from intermediates P R F1 | assessable recall, at n = 2^15
TABLE5 = {"tree": (.677, .579, .623, .503, .375, .425, .731), "dag": (.886, .695, .777, .829, .493, .613, .772),
          "ftree": (.518, .446, .478, .403, .321, .354, .508), "rand": (.670, .501, .569, .632, .415, .492, .533)}

# Table 6 (page 28): step ms Dense, LeWi, ratio; params Dense, LeWi; whole budget s; inference ms; peak RSS MB; GPU step ratio
# Table 6 of the manuscript: Dense step ms, LeWi step ms, overhead in percent, parameters (Dense, LeWi), budget s (Dense, LeWi),
# inference ms over 4,096 examples (Dense, LeWi), peak memory MB (Dense, LeWi); one CPU thread, the optimized implementation
TABLE6 = {"32x4": (1.471, 3.049, 107, 3489, 3900, 97, 211, 0.57, 1.28, 871, 874),
          "64x8": (2.206, 5.301, 140, 30273, 31860, 340, 722, 3.34, 4.56, 917, 930)}

# Table 7 (page 33): (arm key, parameter key) -> (low, high)
TABLE7 = {("dense", "lr"): (8.58e-5, 4.63e-2), ("l1", "lr"): (4.94e-5, 9.85e-2), ("l1", "lam"): (1.66e-9, 1.66e3), ("l2", "lr"): (3.44e-4, 3.98e-2),
          ("l2", "lam"): (5.22e-7, 9.06e1), ("l0hc", "lr"): (1.30e-3, 6.50e-2), ("l0hc", "lam"): (2.75e-9, 1.61e2), ("rigl", "lr"): (1.41e-4, 1.60e-2),
          ("rigl", "density"): (1.58e-2, 1.0), ("bimt", "lr"): (3.85e-4, 1.22e-2), ("bimt", "lam"): (1.25e-8, 7.36e-3), ("fiedler", "lr"): (1.83e-4, 1.68e-2),
          ("fiedler", "delta"): (2.03e-9, 2.84e5), ("erb", "lr"): (2.15e-4, 1.35e-1), ("lewi-v1", "lr"): (9.66e-5, 2.53e-2)}
