"""Appendix: LeWi's sensitivity to its embedding dimension r, its radius multiplier and its initialization scale, and the
connection-density record of every ablation arm (18-main-d8 analysis/ablations).

Two data modes. Full grid (data-*/fig6/sensitivity_full.csv from 18-main-d8/sensitivity-full/tables; the authors' addendum of
2026-09-23), one row of three panels (decision 2026-09-23: the absolute-AULC row is dropped, it repeats the differences
on a scale the two workloads do not share): the paired difference AULC(variant) - AULC(default LeWi) over the canonical grid
(instances 6000-6015, n = 2^9 .. 2^15, 32 trials per condition) with its 95% t interval, on the Depth-4 tree and the Fourier
tree, against three knobs in the method's own units (decision 2026-09-23): the embedding dimension r in {1, 2, 3, 4, 8}
(R^2 = 2 r sigma_e^2 follows r), the connection threshold R^2 in {0.75, 1.125, 1.5, 2.25, 3.0} (the study's multipliers
{0.5, 0.75, 1, 1.5, 2} on the default 1.5 = 2 x 3 x 0.5^2; sigma_e fixed), and the initialization scale sigma_e in
{0.25, 0.5, 1.0} (the multipliers {0.5, 1, 2} on 0.5) with R^2 = 2 r sigma_e^2 following it as in the default construction,
so the initial masks are the default's (17-main-clean/src/abl_methods.py); positive values favour the default. The default's own point is zero with no interval (every instance compared with itself) and is drawn as such;
the adopted value is marked by an orange dashed line (orange: the default LeWi only); the two workloads' points sit side by
side (a 0.07-octave offset) so their intervals do not overlap. Reduced grid
(sensitivity_r.csv, sensitivity_rad.csv, sensitivity_init.csv; drawn only while sensitivity_full.csv is absent): the pilot
sweeps on 6000-6007 at 2^11 / 2^13 / 2^15 with those tables' own sign, AULC(default) - AULC(variant). The rows drawn in the
full-grid mode: fig6_sensitivity.csv beside the figure. The density table (density.csv -> ablation_density_table) is written
in both modes.
"""
from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import paperstyle as ps
from fig6_ablations import NAME
from figstyle import DATA, OUT, save

CELL_COLOR = {"tree8": "#d62728", "ftree8": "#1f77b4"}                    # the reduced-grid figure
STYLE = {"tree8": ("#1a1a1a", "o"), "ftree8": ("#8a8a8a", "s")}           # full grid: black circles, grey squares (blue is mask-only in Figure 6)
DEFAULT_COLOR = ps.COLOR[ps.LEWI]                                         # orange: the default LeWi only
DEFAULT_LINE = (0, (4, 2))
R2_DEFAULT, SIGMA_E = 1.5, 0.5                                            # LeWi's R^2 = 2 r sigma_e^2 at r = 3, sigma_e = 0.5
SWEEPS = (("r", "embedding dimension $r$"), ("radius", "connection threshold $R^2$"), ("init_scale", r"initialization scale $\sigma_e$"))
UNIT = {"r": 1.0, "radius": R2_DEFAULT, "init_scale": SIGMA_E}           # the study's values (r, multipliers) in the method's units
DODGE = 0.07                                                              # octaves between the two workloads' points


def _axes_style(ax, values) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(values)                                 # every swept value, the default's included, as a plain label
    ax.set_xticklabels([f"{v:g}" for v in values])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)


def sweep(ax_top, ax_bot, df: pd.DataFrame, key: str, xlabel: str) -> None:
    """Reduced grid: the earlier tables (their sign: AULC(default) - AULC(variant))."""
    for cell, color in CELL_COLOR.items():
        sub = df[df.cell == cell].sort_values(key)
        if sub.empty:
            continue
        ax_top.errorbar(sub[key], sub["mean"], yerr=[sub["mean"] - sub.ci95_lo, sub.ci95_hi - sub["mean"]], fmt="o-", color=color, markersize=3.5, capsize=2, linewidth=0.9, label=NAME[cell])
        v = sub[sub.arm != ps.LEWI]
        ax_bot.errorbar(v[key], v.diff_mean, yerr=[v.diff_mean - v.diff_lo, v.diff_hi - v.diff_mean], fmt="o", color=color, markersize=3.5, capsize=2, linewidth=0.9)
    for ax in (ax_top, ax_bot):
        _axes_style(ax, sorted(df[key].unique()))
    ax_bot.axhline(0, color="#888", linewidth=0.7)
    ax_top.set_ylabel("AULC (reduced grid)", fontsize=8)
    ax_bot.set_ylabel("AULC(default) $-$ AULC(variant)", fontsize=8)
    ax_bot.set_xlabel(xlabel)


def sweep_full(ax, df: pd.DataFrame, xlabel: str) -> None:
    """Full grid: one sweep's paired differences (the sign of the addendum: AULC(variant) - AULC(default))."""
    default = df[df.is_default.astype(bool)]
    assert (default[["diff_mean", "diff_lo", "diff_hi"]] == 0).all().all(), "the default's paired difference is not identically zero"
    default_value = float(default.value.iloc[0])
    for k, (cell, (color, marker)) in enumerate(STYLE.items()):
        sub = df[df.cell == cell].sort_values("value")
        if sub.empty:
            continue
        shift = 2 ** ((k - 0.5) * DODGE)
        v = sub[~sub.is_default.astype(bool)]
        ax.errorbar(v.value * shift, v.diff_mean, yerr=[v.diff_mean - v.diff_lo, v.diff_hi - v.diff_mean], fmt=marker, color=color,
                    markersize=3.5, capsize=2, linewidth=0.9, zorder=3)
        d = sub[sub.is_default.astype(bool)]
        ax.plot(d.value * shift, d.diff_mean, marker=marker, color=color, markersize=3.5, linestyle="none", zorder=4)
    ax.axvline(default_value, color=DEFAULT_COLOR, linewidth=1.0, linestyle=DEFAULT_LINE, zorder=0)
    ax.axhline(0, color="#888", linewidth=0.7, zorder=1)
    _axes_style(ax, sorted(df.value.unique()))
    if xlabel != SWEEPS[0][1]:                                            # R^2 and sigma_e as decimals: 3.0, 1.0
        ax.set_xticklabels([f"{v:g}" if "." in f"{v:g}" else f"{v:g}.0" for v in sorted(df.value.unique())])
    ax.grid(False)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(labelsize=7.5)
    ax.set_xlabel(xlabel, fontsize=8)


def draw_full(full: pd.DataFrame) -> None:
    sweeps = [(k, label) for k, label in SWEEPS if (full.sweep == k).any()]
    fig, axes = plt.subplots(1, len(sweeps), figsize=(5.5, 2.35), sharey=True, squeeze=False)
    for ax, (key, xlabel) in zip(axes[0], sweeps):
        sweep_full(ax, full[full.sweep == key].assign(value=lambda d, u=UNIT[key]: d.value * u), xlabel)
    axes[0, 0].set_ylabel(r"$\mathrm{AULC}(\mathrm{variant})-\mathrm{AULC}(\mathrm{default})$", fontsize=8)
    handles = [Line2D([], [], marker=m, color=c, markersize=3.5, linewidth=0, label=NAME[cell]) for cell, (c, m) in STYLE.items()]
    handles.append(Line2D([], [], color=DEFAULT_COLOR, linewidth=1.0, linestyle=DEFAULT_LINE, label="adopted value"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5, frameon=False, handlelength=1.8, columnspacing=1.6)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.97, bottom=0.31, wspace=0.1)
    save(fig, "fig6_sensitivity")
    full.to_csv(OUT / "fig6_sensitivity.csv", index=False)
    print(f"[fig6_sensitivity] full grid: {len(full)} rows, {len(sweeps)} sweeps")


def draw_reduced(r: pd.DataFrame, rad: pd.DataFrame, ini: pd.DataFrame) -> None:
    if r.empty and rad.empty:
        print("[fig6_sensitivity] no sensitivity tables yet: nothing drawn")
        return
    ncol = 3 if not ini.empty else 2
    fig, axes = plt.subplots(2, ncol, figsize=(5.5 if ncol == 2 else 7.0, 3.8), sharex="col")
    if not r.empty:
        sweep(axes[0, 0], axes[1, 0], r, "r", "position dimension $r$ (default: 3)")
    if not rad.empty:
        sweep(axes[0, 1], axes[1, 1], rad, "radius", "radius multiplier of $R^2$ (default: 1)")
    if not ini.empty:
        sweep(axes[0, 2], axes[1, 2], ini, "init_scale", "initialization scale (default: 1)")
    for j in range(1, ncol):
        axes[0, j].set_ylabel("")
        axes[1, j].set_ylabel("")
    fig.legend(handles=[Line2D([], [], marker="o", color=c, markersize=3.5, linewidth=0.9, label=NAME[cell]) for cell, c in CELL_COLOR.items()],
               loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.19, hspace=0.15, wspace=0.3)
    save(fig, "fig6_sensitivity")


def density_table(dens: pd.DataFrame) -> None:
    if dens.empty:
        return
    head = "| panel | workload | arm | winners | initial density | final density | dead hidden units (share) | dead winners | readout redraws | spread over the 32 learning-rate trials (log10 test NMSE) |"
    rows = [head, "|---|---|---|---|---|---|---|---|---|---|"]
    tex = ["\\begin{tabular}{lllrrrrrrr}", "\\toprule",
           "panel & workload & arm & winners & initial density & final density & dead hidden units & dead winners & redraws & lr-trial spread \\\\", "\\midrule"]
    for _, x in dens.iterrows():
        cells = [x.panel, x.workload, x.label, f"{int(x.winners)}", f"{x.init_density:.3f}", f"{x.final_density:.3f}", f"{x.dead_hidden_share:.3f}", f"{int(x.dead_winners)}",
                 f"{x.redraws_mean:.2f}", f"{x.lr_trial_spread_log10_test:.3f}"]
        rows.append("| " + " | ".join(cells) + " |")
        tex.append(" & ".join(c.replace("_", "\\_").replace("²", "$^2$").replace("×", "$\\times$") for c in cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    note = ("One scope for both densities: every transition, the readout included, from each winner's own init.pt and best.pt. For the soft-only arm the final density and the "
            "dead units describe the would-be mask, which its forward pass does not apply. The 32 trials of a condition share every seed and differ in the learning rate only.")
    (OUT / "ablation_density_table.md").write_text("# Connection density and initialization record of the ablation and sensitivity arms\n\n" + "\n".join(rows) + "\n\n" + note + "\n")
    (OUT / "ablation_density_table.tex").write_text("\n".join(tex) + "\n")
    print(f"[fig6_sensitivity] density table: {len(dens)} rows")


def main() -> None:
    ps.apply_rc()

    def _read(name):
        f = DATA / "fig6" / name
        return pd.read_csv(f) if f.exists() and f.stat().st_size > 5 else pd.DataFrame()
    full = _read("sensitivity_full.csv")
    if not full.empty:
        draw_full(full)
    else:
        draw_reduced(_read("sensitivity_r.csv"), _read("sensitivity_rad.csv"), _read("sensitivity_init.csv"))
    density_table(_read("density.csv"))


if __name__ == "__main__":
    main()
