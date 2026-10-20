# Reproduction map

Every experimental figure and table of the manuscript: the command that regenerates it from the packaged records, the
configuration behind the underlying runs, the supplied record, and the expected output. `FIG` =
`studies/communication/paper/figures`, `S17` = `studies/main-experiments/17-main-clean`, `S18` =
`studies/main-experiments/18-main-d8`, `S16` = `studies/main-experiments/16-main-final`. `bash scripts/rebuild_figures.sh`
runs every row; `python scripts/verify_results.py` recomputes every number (its section letter is given per row).

The package was made for the submitted draft. Three items differ in the arXiv version (Table 6, the colour rule of the
learned-embedding figures, the depth-5 tree percentage): see README.md, "Differences from the arXiv version".

| item | command | configuration | supplied record | expected output |
|---|---|---|---|---|
| Figure 1 (learned embedding geometry, depth-4 Fourier tree) | `rebuild_figures.sh` (design/teaser-paper: the Fourier-tree panel) | the selected LeWi run of `ftree8`, instance 6003, n = 2^15 (`S18/ledger/d8/winners.csv`); Figure 1 is this network under another camera and layout (its design settings are not included) | `FIG/design/teaser-paper/data/ftree_f6003_n32768__lewi-v1-rgm.npz` (the run's embeddings and masks) | `FIG/out-paper/d8/embedding_ftree.pdf` |
| Figure 2 (method schematic) | none (drawing) | - | - | - |
| Figure 3 (computation graphs) | `rebuild_figures.sh graph_templates` | the graph templates of `src/csfrbench/tasks` | drawn from the package | `out-paper/d8/graph_templates.pdf` |
| Figure 4 (learning curves, six settings) | `rebuild_figures.sh sample_efficiency` | panel `main8b` (+ `go8b`), selection rule | `FIG/data-paper-d8/sample_efficiency/curves.csv` from `S18/ledger/d8/winners.csv` | `out-paper/d8/sample_efficiency.pdf`; verify A, B |
| Figure 5 (linear probes and counterfactual interventions, n = 2^14) | `rebuild_figures.sh learned_computational_structure_n16384` | `m17_posthoc.py probes-* / das-*` on the selected `tree8` runs; DAS: `src/csfrbench/das_*.py` | `data-paper-d8/learned_structure/probe_values.csv`, `data-paper-d8/causal_nopatch/cells.csv` | `out-paper/d8/learned_computational_structure_n16384.{pdf,csv}`; verify D |
| Figure 6 (nuisance inputs, wiring alignment, mechanism ablations) | `rebuild_figures.sh fig6` | (a) panels `main8b` vs `main32s`; (b) `S18/structure/src/sd_units.py`; (c) panel `abl8-in5` (`abl_methods.py`), `S18/src/ablations.py` | `data-paper-d8/fig6/feature_selection.csv`, `data-paper-d8/structure/fig6b.csv`, `data-paper-d8/fig6/ablations.csv` | `out-paper/d8/fig6.pdf`; verify B, C, E |
| Figure 7 (world-model heatmap) | `rebuild_figures.sh world_model_grid_compact` | `S16/src/swm16.py` + the three layers | `data-paper-d8/world_model_transfer_all/paired.csv` from `S16/external/analysis/instances.csv` | `out-paper/d8/world_model_grid_compact.pdf`; verify H |
| Figure 8 (sampled computation graphs) | `rebuild_figures.sh random_graphs` | `src/csfrbench/tasks/rand.py` | drawn from the sampler | `out-paper/d8/random_graphs.pdf` |
| Figure 9 (local-function fitting capacity) | `rebuild_figures.sh local_fit` | the development capacity grid | `data-paper-d8/local_fit/a4_grid.csv` | `out-paper/d8/local_fit.pdf` |
| Figure 10 (GraphOracle masks) | `rebuild_figures.sh graph_oracle_masks` | `src/csfrbench/methods/graph_oracle.py` | `data-paper-d8/graph_oracle_masks/expected_sha.json` (hash check) | `out-paper/d8/graph_oracle_masks.pdf` |
| Figure 11 (all learning curves, d = 8) | `rebuild_figures.sh learning_curves_all` | as Figure 4 | as Figure 4 | `out-paper/d8/learning_curves_all.pdf`; verify A, B |
| Figure 12 (paired AULC, d = 8) | `rebuild_figures.sh sample_efficiency_aulc` | as Figure 4 | as Figure 4 | `out-paper/d8/sample_efficiency_aulc.pdf`; verify B |
| Figure 13 (diagnostics, additional baselines) | `rebuild_figures.sh learned_computational_structure_baselines_n16384` | as Figure 5 (L0, RigL, BIMT) | as Figure 5 | `out-paper/d8/learned_computational_structure_baselines_n16384.{pdf,csv}`; verify D |
| Figure 14 (diagnostics across training-set size) | `rebuild_figures.sh representation_diagnostics_across_n` | as Figure 5 | `probe_values.csv`, `causal_nopatch/scores.csv` | `out-paper/d8/representation_diagnostics_across_n.{pdf,csv}`; verify D |
| Figure 15 (all learning curves, nuisance inputs) | `rebuild_figures.sh` (d = 32 block) | panels `main32s`, `main32t` | `FIG/data-paper-d32/sample_efficiency/curves.csv` from `S18/ledger/d32s/winners.csv` | `out-paper/d32/learning_curves_all.pdf`; verify B, C |
| Figure 16 (paired AULC, nuisance inputs) | `rebuild_figures.sh` (d = 32 block) | as Figure 15 | as Figure 15 | `out-paper/d32/sample_efficiency_aulc.pdf`; verify B, C |
| Figure 17 (target and extracted module graphs) | `rebuild_figures.sh structure_modules` | `S18/structure/src/sd_units.py` on the selected LeWi runs at n = 2^15 | `data-paper-d8/structure/examples.json` + `examples/*.json` | `out-paper/d8/structure_modules.pdf` |
| Figure 18 (wiring alignment across training-set size) | `rebuild_figures.sh structure_alignment_vs_n` | as Figure 17 | `data-paper-d8/structure/alignment_vs_n.csv` | `out-paper/d8/structure_alignment_vs_n.pdf`; verify E |
| Figure 19 (learned neuron embeddings across settings) | `rebuild_figures.sh` (design/teaser-paper, then `embeddings_stack`) | selected LeWi runs at n = 2^15 | `FIG/design/teaser-paper/data/*.npz`, `choices.json` | `out-paper/d8/embeddings_main.pdf` |
| Figure 20 (learned connection density) | `rebuild_figures.sh connection_density` | `S18/density/src/density.py` on the selected LeWi runs | `S18/density/tables/{by_size,per_run}.csv` | `out-paper/d8/connection_density.{pdf,csv}`; verify F |
| Figure 21 (sensitivity to the geometric parameterization) | `rebuild_figures.sh fig6_sensitivity` | panels `abl8r-in5`, `abl8rad-in5`, `abl8init-in5`, `abl8seed-in5`, `lms8`, `lms5` | `data-paper-d8/fig6/sensitivity_full.csv` | `out-paper/d8/fig6_sensitivity.{pdf,csv}` |
| Figure 22 (complete world-model comparisons) | `rebuild_figures.sh world_model_grid` | as Figure 7 | as Figure 7 | `out-paper/d8/world_model_grid_full.pdf`; verify H |
| Figure 23 (world-model learning curves) | `rebuild_figures.sh world_model_transfer_all` | as Figure 7 | `world_model_transfer_all/{cells,paired}.csv` | `out-paper/d8/world_model_transfer_all.pdf`; verify H |
| Table 1 (evaluation settings) | none (prose) | `S17/src/m17.py` CELLS; `src/csfrbench/tasks` | - | verify K (instances, grids) |
| Table 2 (search spaces, CSFRBench) | none (record) | the frozen ranges `S17/manifests/search_spaces.json` (sets `p8b` for 32x4, `p0` for the depth-5 tree, the appendix sets) | `FIG/out-paper/d8/search_space_table.csv` | verify I |
| Table 3 (mean AULC, six fixed settings) | `rebuild_figures.sh aulc_table` | as Figure 4 | as Figure 4; `S18/analysis/d8/contract/aulc.csv` | `out-paper/d8/aulc_table.{md,tex}`; verify B (every cell) |
| Table 4 (depth-5 tree, sampled depth-4 DAGs) | `rebuild_figures.sh additional_structures_table` | panels `main` (tree16), `rand8b` | as Figure 4 | `out-paper/d8/additional_structures_table.{md,tex}`; verify B (every cell) |
| Table 5 (wiring alignment at n = 2^15) | `rebuild_figures.sh wiring_alignment_table` | as Figure 17 | `data-paper-d8/structure/wiring_alignment.csv` (= `S18/structure/tables/wiring_alignment.csv`) | `out-paper/d8/wiring_alignment_table.{md,tex}`; verify E |
| Table 6 (computational cost) | none (record) | `S18/overhead-fast/src/fast_lewi.py`, one CPU thread, nine paired timing blocks | `S18/overhead-fast/tables/overhead_fast.csv` (rows `impl = exact`, `device = cpu`, `threads = 1`, arms `noreg-exact`, `lewi-k1-exact`) | verify G (every cell of the table, the 107-140% overhead) |
| Table 7 (search spaces, world models) | `rebuild_figures.sh world_model_hpo_table` | `09-swm-lewi/results/spaces_final_v2.json`, `S18/swm-lrx/manifests/spaces_lrx.json` | `FIG/out-paper/d8/world_model_hpo_table.csv` | `out-paper/d8/world_model_hpo_table.{md,tex,csv}`; verify H |
| main-text numbers (20% / 34% / 27% / 31%, 16 of 16, 32 of 32, 51% and 26% gap shares, edge-F1 ratio, 99-6-14, 16 of 17, 107-140%) | `python scripts/verify_results.py` | - | `S18/analysis/d8/contract`, `S18/structure/tables`, `data-paper-d8/world_model_transfer_all`, `overhead_fast.csv` | verify B, E, G, H |
| Appendix C.12 (boundary audit) | `python scripts/verify_results.py` | the frozen ranges | `S18/analysis/hpo_boundary/boundary_occupancy{,_d32s}.csv` | verify J |
| Appendix D.5 (densities) | `python scripts/verify_results.py` | `S18/density/src/density.py` | `S18/density/tables/{summary,by_size}.csv` | verify F; `rebuild_figures.sh connection_density_table` |
