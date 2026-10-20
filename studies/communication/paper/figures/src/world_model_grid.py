"""Figure: where LeWi improves world-model sample efficiency - every counted task against every baseline.

One builder, two stems (decision 2026-09-23, replacing the plate of 2026-09-14):
  world_model_grid       main text: the heat map alone against the seven principal baselines - short rows, compact
                         group names, the colour bar at the right (decision: the neuron-position reference only in
                         the appendix, the scale from the seven rows drawn)
  world_model_grid_full  appendix: every baseline, neuron positions included, with the values printed and LeWi's
                         wins-ties-losses per baseline
CAPTION and CAPTION_FULL below are filled with the numbers of the build and printed by main().

Data: data/world_model_transfer_all/paired.csv, the `aulc` rows that src/harvest_swm16.py writes from the SWM-16 campaign
(16-main-final/external: 19 tasks x 7 sizes x 9 arms x 16 paired seeds, 256x4 backbone, 128,000 steps; the 17 tasks of its
task gate are drawn) in its ADOPTED form: the record with the 30 final seeds the input rule in5 re-ran (decision) and
the 480 finals of the 30 cells the validation-only second-stage re-selection changed (decision) substituted, the
paired contrasts recomputed from that per-seed table as the record computes them. swm16.json names both adoptions and any
re-run still pending; the build refuses a paired.csv without them. The Dense and l1 re-search with the learning-rate lower
bound one decade lower (18-main-d8/swm-lrx) is adopted as well (decision 2026-09-24): its 3,808 finals are the record's
Dense and l1 arms on the counted tasks (swm16.json lrx_adopted). The verdict changes against the record as run are printed and put in
CAPTION_FULL; the filled captions are written beside the figures as <stem>.caption.txt. CSFR_SWM_SEALED=1 draws the record as run (stems with the suffix _sealed; provenance only), CSFR_SWM_TRIAL=1
the trial table of 2026-09-18 (_trial).

Design: a linear diverging scale symmetric about zero, vmin = -vmax with vmax the largest |paired mean| drawn (no separate
limits for the two signs); ColorBrewer RdBu reversed (Brewer 2003; colour-blind safe), blue = LeWi better; a small centred
dot for an interval that contains zero; real gaps between the task groups; no key prose in the main plate (the appendix
plate keys its dot on the bar's line, decision 2026-09-25). Built at the include
width (\\linewidth = 5.5 in), so the type prints at the sizes below.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

import paperstyle as ps
from figstyle import DATA, OUT, save

INK, MUTED, SEAM_COLOUR, DOT = "#1a1a1a", "#6b6b6b", "#d0d0d0", "#262626"
CMAP = plt.get_cmap("RdBu_r")
# rows: the paper's baseline order, top to bottom
BASELINES = [("dense", "Dense"), ("l1", r"$\ell_1$"), ("l2", r"$\ell_2$"), ("l0hc", r"$L_0$"),
             ("rigl", "RigL"), ("bimt", "BIMT"), ("fiedler", "Fiedler"), ("erb", "Neuron positions")]
MAIN_ROWS = [k for k, (arm, _l) in enumerate(BASELINES) if arm != "erb"]      # the main text's seven principal baselines
# columns: the counted suite, grouped by environment type and ordered within a
# group by state dimension (decision 2026-09-23), ties alphabetical; the obs_dim of each task's dataset
# (09-swm-lewi/data/<task>_xl.meta.json; PushT's 7 is 8 after featurization)
STATE_DIM = {"pendulum": 3, "pendulumctl": 3, "cartpole": 5, "acrobot": 6, "reacher": 6, "ballincup": 8, "finger": 12, "hopper": 15,
             "cheetah": 17, "walker": 24, "quadruped": 78, "ogbmaze": 2, "pointmaze": 2, "tworoom": 10, "pusht": 7, "manipulator": 44,
             "piecewise": 4}
GROUPS = [(g, sorted(ts, key=lambda t: STATE_DIM[t[0]])) for g, ts in (
    ("Control", [("acrobot", "Acrobot"), ("ballincup", "Ball in cup"), ("cartpole", "Cartpole"), ("cheetah", "Cheetah"),
                 ("finger", "Finger"), ("hopper", "Hopper"), ("pendulum", "Pendulum (DM)"), ("pendulumctl", "Pendulum (Gym)"),
                 ("quadruped", "Quadruped"), ("reacher", "Reacher"), ("walker", "Walker")]),
    ("Navigation", [("ogbmaze", "OGBench maze"), ("pointmaze", "Point maze"), ("tworoom", "Two-room")]),
    ("Manipulation", [("manipulator", "Manipulator"), ("pusht", "PushT")]),
    ("Synthetic", [("piecewise", "Piecewise")]))]
TASKS = [t for _g, ts in GROUPS for t in ts]
GROUP_OF = [g for g, (_name, ts) in enumerate(GROUPS) for _t in ts]
# geometry in inches
WIDTH, GAP, HEAD = 5.5, 0.04, 0.2                  # figure width, gap between groups, room for group names
LEFT_MAIN, LEFT_FULL = 0.5, 0.88                   # the row-name margin: the main plate's longest name is Fiedler's
ROW_MAIN, ROW_FULL = 0.15, 0.21
BAR_BLOCK, BAR_GAP, BAR_W = 0.48, 0.12, 0.075      # main: the colour bar's column at the right of the body
TALLY_BLOCK = 0.46                                 # appendix: the wins-ties-losses column
ROW_SIZE, TASK_SIZE, GROUP_SIZE, BAR_SIZE, VALUE_SIZE = 7.5, 7.0, 7.0, 6.5, 6.0
DOT_SIZE = {False: 1.8, True: 1.5}                 # points: the main plate's dot, the appendix plate's (a cell's corner)

TRIAL = os.environ.get("CSFR_SWM_TRIAL") == "1"
SEALED = os.environ.get("CSFR_SWM_SEALED") == "1"
_SUFFIX = "_trial" if TRIAL else ("_sealed" if SEALED else "")

CAPTION = (
    "LeWi improves rollout sample efficiency on the 17 counted world-model tasks. Each cell is the paired difference "
    "AULC(LeWi) $-$ AULC(baseline) for one task (column) and one of the seven principal baselines (row); blue: LeWi "
    "better, red: the baseline better. Against them LeWi wins {win} of the {n} comparisons, ties {tie} and loses {loss}. "
    "The AULC is the area under the $\\log_{{10}}$ 16-step open-loop rollout NMSE over $\\log_2 n$, divided by "
    "the $\\log_2 n$ range (a trapezoidal average over the seven training-set sizes); the difference is taken within "
    "each of the 16 paired seeds and the cell shows the mean. The colour scale is linear and symmetric about zero, "
    "from $-{vmax:.2f}$ to ${vmax:.2f}$, the largest absolute difference drawn. A dot marks the {tie} comparisons "
    "whose paired 95\\% Student-$t$ interval contains zero. Tasks are grouped by environment type and ordered within "
    "each group by state dimension. Pendulum (DM): DeepMind Control Suite; Pendulum (Gym): Gymnasium. The neuron-position reference ({erb}) and every value: Figure~\\ref{{fig:world-model-grid-full}}.")
CAPTION_FULL = (
    "World-model comparisons with values and tallies. The comparisons of Figure~\\ref{{fig:world-model-grid}} and those "
    "against the neuron-position reference, with each paired mean printed ($\\log_{{10}}$ units; a dot: the paired 95\\% interval contains zero) and, at the right, LeWi's wins, "
    "ties and losses against each baseline over the 17 counted tasks (a win or a loss is an interval that excludes "
    "zero); the tasks in the order of Figure~\\ref{{fig:world-model-grid}}, by environment type and, within a group, "
    "by state dimension. In total LeWi wins {win} of {n} comparisons, ties {tie} and loses {loss}. The colour scale is symmetric about "
    "zero to $\\pm{vmax:.2f}$, the largest absolute difference in this figure. The record is the campaign's adopted form: "
    "{adopted} {changes}")


def load(suffix: str) -> pd.DataFrame | None:
    """The campaign's paired AULC rows, one per task and baseline; None in a data folder of an older world-model lane."""
    path = DATA / f"world_model_transfer_all/paired{suffix}.csv"
    paired = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["n", "complete", "reference", "task"])
    aulc = paired[(paired.n == "aulc") & paired.complete.astype(bool)].set_index(["reference", "task"])
    missing = [(b, t) for b, _l in BASELINES for t, _n in TASKS if (b, t) not in aulc.index]
    if missing:
        print(f"[world_model_grid] skipped: {DATA.name} has no SWM-16 paired rows ({len(missing)} of 136 cells absent)")
        return None
    return aulc


def collect(aulc: pd.DataFrame):
    """The paired means (rows x columns), the verdicts, and LeWi's win / tie / loss counts per baseline."""
    value = np.array([[aulc.loc[(b, t)].dlog10_mean for t, _n in TASKS] for b, _l in BASELINES])
    lo = np.array([[aulc.loc[(b, t)].dlog10_lo for t, _n in TASKS] for b, _l in BASELINES])
    hi = np.array([[aulc.loc[(b, t)].dlog10_hi for t, _n in TASKS] for b, _l in BASELINES])
    verdict = np.where(hi < 0, "win", np.where(lo > 0, "loss", "tie"))
    recorded = np.array([[aulc.loc[(b, t)].verdict for t, _n in TASKS] for b, _l in BASELINES])
    assert (verdict == recorded).all(), "the intervals and the recorded verdicts disagree"
    counts = [{v: int((verdict[i] == v).sum()) for v in ("win", "tie", "loss")} for i in range(len(BASELINES))]
    return value, verdict, counts


def number(v: float, fmt: str = "g") -> str:
    return format(v, fmt).replace("-", "−")


def is_dark(rgba) -> bool:
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgba[:3]]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2] < 0.3


def draw(value: np.ndarray, verdict: np.ndarray, counts: list[dict], labels: list[str], vmax: float, full: bool) -> plt.Figure:
    rows, cols = value.shape
    row_h = ROW_FULL if full else ROW_MAIN
    right = TALLY_BLOCK if full else BAR_BLOCK
    left = LEFT_FULL if full else LEFT_MAIN
    cell = (WIDTH - left - right - GAP * (len(GROUPS) - 1)) / cols
    xs = np.array([j * cell + GAP * GROUP_OF[j] for j in range(cols)])      # left edge of every column, from the body's left
    body_w, body_h = xs[-1] + cell, rows * row_h
    foot = 0.62 + (0.34 if full else 0.0)                                 # the task names (and the appendix's bar) below
    height = HEAD + body_h + foot
    fig = plt.figure(figsize=(WIDTH, height))
    renderer = fig.canvas.get_renderer()
    width = lambda t: t.get_window_extent(renderer).width / fig.dpi      # a text's printed width, inches
    ax = fig.add_axes([left / WIDTH, foot / height, body_w / WIDTH, body_h / height])
    ax.set_xlim(0, body_w)
    ax.set_ylim(body_h, 0)                                                # inches, the first baseline on top
    ax.axis("off")
    norm = Normalize(-vmax, vmax)
    for i in range(rows):
        for j in range(cols):
            colour = CMAP(norm(value[i, j]))
            ax.add_patch(Rectangle((xs[j], i * row_h), cell, row_h, facecolor=colour, edgecolor=SEAM_COLOUR, linewidth=0.3))
            if full:
                ax.text(xs[j] + cell / 2, (i + 0.5) * row_h, number(value[i, j], ".2f"), ha="center", va="center",
                        fontsize=VALUE_SIZE, color="white" if is_dark(colour) else INK, clip_on=False)
            if verdict[i, j] == "tie":                                    # an interval that contains zero
                x, y = (xs[j] + 0.9 * cell, (i + 0.16) * row_h) if full else (xs[j] + cell / 2, (i + 0.5) * row_h)
                ax.plot(x, y, marker="o", markersize=DOT_SIZE[full], color=DOT, markeredgewidth=0, linestyle="none")
    for i, label in enumerate(labels):
        ax.text(-0.06, (i + 0.5) * row_h, label, ha="right", va="center", fontsize=ROW_SIZE, color=INK, clip_on=False)
    for j, (_task, name) in enumerate(TASKS):
        ax.text(xs[j] + cell / 2 + 0.03, body_h + 0.035, name, rotation=45, rotation_mode="anchor", ha="right", va="top",
                fontsize=TASK_SIZE, color=INK, clip_on=False)

    # group names over their columns, pushed apart where a name is wider than its group
    heads = []
    for g, (name, _tasks) in enumerate(GROUPS):
        members = [j for j in range(cols) if GROUP_OF[j] == g]
        heads.append(ax.text((xs[members[0]] + xs[members[-1]] + cell) / 2, -0.04, name, ha="center", va="bottom",
                             fontsize=GROUP_SIZE, color=MUTED, clip_on=False))
    limit = body_w + (0.06 if full else right - 0.02)                   # the appendix keeps the tally header free
    for a, b in zip(heads, heads[1:]):                                   # left to right: no two names closer than 0.05 in
        push = (a.get_position()[0] + width(a) / 2 + 0.05) - (b.get_position()[0] - width(b) / 2)
        if push > 0:
            b.set_x(b.get_position()[0] + push)
    over = heads[-1].get_position()[0] + width(heads[-1]) / 2 - limit
    for k in range(len(heads) - 1, -1, -1):                              # right to left: the last name within the limit
        if over <= 0:
            break
        heads[k].set_x(heads[k].get_position()[0] - over)
        if k:
            a, b = heads[k - 1], heads[k]
            over = (a.get_position()[0] + width(a) / 2 + 0.05) - (b.get_position()[0] - width(b) / 2)

    if full:                                                             # LeWi's wins - ties - losses, in aligned slots
        slots = []
        for s in ("00", "–", "0", "–", "0"):
            probe = ax.text(0, 0, s, fontsize=ROW_SIZE)
            slots.append(width(probe))
            probe.remove()
        x0 = body_w + 0.12

        def tally(pieces: list[str], y: float, colour: str, va: str) -> None:
            x = x0
            for piece, slot in zip(pieces, slots):
                ax.text(x + slot / 2, y, piece, ha="center", va=va, fontsize=ROW_SIZE if va == "center" else GROUP_SIZE,
                        color=colour, clip_on=False)
                x += slot + 0.012

        tally(["W", "–", "T", "–", "L"], -0.04, MUTED, "bottom")
        for i, c in enumerate(counts):
            tally([str(c["win"]), "–", str(c["tie"]), "–", str(c["loss"])], (i + 0.5) * row_h, INK, "center")

    ticks = [t for t in MaxNLocator(nbins=8, symmetric=True, steps=[1, 2, 2.5, 5, 10]).tick_values(-vmax, vmax)
             if abs(t) <= vmax + 1e-9]
    ramp = np.linspace(-vmax, vmax, 512)
    if full:                                                             # a short horizontal bar under the task names
        bar_w, bar_h, bar_y = 1.9, 0.065, 0.13
        x_bar = left + body_w / 2 - bar_w / 2
        cax = fig.add_axes([x_bar / WIDTH, bar_y / height, bar_w / WIDTH, bar_h / height])
        cax.imshow(ramp[None, :], aspect="auto", cmap=CMAP, norm=norm, extent=[-vmax, vmax, 0, 1], interpolation="nearest")
        cax.set_yticks([])
        cax.set_xticks(ticks)
        cax.set_xticklabels([number(t) for t in ticks], fontsize=BAR_SIZE, color=INK)
        cax.tick_params(axis="x", length=1.8, width=0.4, pad=1.2, color=MUTED)
        fig.text((x_bar - 0.08) / WIDTH, (bar_y + bar_h / 2) / height, r"$\Delta$AULC", ha="right", va="center",
                 fontsize=GROUP_SIZE, color=INK)
        key_x, key_y = x_bar + bar_w + 0.3, bar_y + bar_h / 2          # the key of the dot, on the bar's line
        fig.add_artist(Line2D([key_x / WIDTH], [key_y / height], transform=fig.transFigure, marker="o",
                              markersize=DOT_SIZE[full], color=DOT, markeredgewidth=0, linestyle="none"))
        fig.text((key_x + 0.05) / WIDTH, key_y / height, "95% CI includes 0", ha="left", va="center", fontsize=GROUP_SIZE,
                 color=INK)
    else:                                                                # a slim vertical bar beside the body
        cax = fig.add_axes([(left + body_w + BAR_GAP) / WIDTH, foot / height, BAR_W / WIDTH, body_h / height])
        cax.imshow(ramp[:, None], aspect="auto", origin="lower", cmap=CMAP, norm=norm, extent=[0, 1, -vmax, vmax],
                   interpolation="nearest")
        cax.set_xticks([])
        cax.yaxis.tick_right()
        cax.set_yticks(ticks)
        cax.set_yticklabels([number(t) for t in ticks], fontsize=BAR_SIZE, color=INK)
        cax.tick_params(axis="y", length=1.8, width=0.4, pad=1.2, color=MUTED)
        cax.yaxis.set_label_position("right")
        cax.set_ylabel(r"$\Delta$AULC", fontsize=GROUP_SIZE, color=INK, labelpad=2)
    for side in cax.spines.values():
        side.set_linewidth(0.4)
        side.set_edgecolor(MUTED)
    return fig


def main() -> None:
    ps.apply_rc()
    aulc = load(_SUFFIX)
    if aulc is None:
        return
    value, verdict, counts = collect(aulc)
    tally = lambda k: f"{counts[k]['win']}-{counts[k]['tie']}-{counts[k]['loss']}"
    adopted, changes = "the record as run (provenance only).", ""
    if not (SEALED or TRIAL):
        meta = json.loads((DATA / "world_model_transfer_all/swm16.json").read_text())
        assert meta.get("in5_adopted") and meta.get("resel_adopted") and meta.get("lrx_adopted"), "paired.csv is not the adopted record: re-run harvest_swm16.py"
        pending = meta["resel_adopted"].get("pending")
        lrx = meta["lrx_adopted"]
        adopted = (f"the record as run with the {meta['in5_adopted']['seeds']} final seeds the input rule re-ran, the "
                   f"{meta['resel_adopted']['finals']} finals of the {len(meta['resel_adopted']['cells'])} cells the "
                   f"validation-only re-selection changed and, for Dense and $\\ell_1$, the {lrx['finals']:,} finals of the re-search with the "
                   f"learning-rate lower bound one decade lower ({lrx['cells']} cells) substituted" + (f" (pending: {pending})" if pending else "") + ".")
        sealed = load("_sealed")
        if sealed is None:
            raise SystemExit("paired_sealed.csv missing: run harvest_swm16.py with CSFR_SWM_SEALED=1 first")
        _v, verdict_sealed, counts_sealed = collect(sealed)
        diff = [(TASKS[c][1], BASELINES[r][1], str(verdict_sealed[r, c]), str(verdict[r, c])) for r, c in zip(*np.nonzero(verdict != verdict_sealed))]
        for name, rows in (("seven principal baselines", MAIN_ROWS), ("all eight", range(len(BASELINES)))):
            print(f"[world_model_grid] {name}: sealed " + "-".join(str(int((verdict_sealed[list(rows)] == v).sum())) for v in ("win", "tie", "loss"))
                  + " -> adopted " + "-".join(str(int((verdict[list(rows)] == v).sum())) for v in ("win", "tie", "loss")))
        changes = ("Against the record as run " + (", ".join(f"{b} on {t}: {s} to {a}" for t, b, s, a in diff) if diff else "no verdict changes")
                   + "; tallies " + "; ".join(f"{BASELINES[k][1]} {'-'.join(str(counts_sealed[k][v]) for v in ('win', 'tie', 'loss'))} to "
                                            f"{'-'.join(str(counts[k][v]) for v in ('win', 'tie', 'loss'))}"
                                            for k in range(len(BASELINES)) if counts[k] != counts_sealed[k]) + ".")
        print(f"[world_model_grid] verdict changes against the record as run: {diff}")
    for k, (_arm, label) in enumerate(BASELINES):
        print(f"[world_model_grid] {label}: {tally(k)} (win-tie-loss); range {value[k].min():+.3f} .. {value[k].max():+.3f}")
    fill = {}
    for full, rows in ((False, MAIN_ROWS), (True, list(range(len(BASELINES))))):
        v, d = value[rows], verdict[rows]
        vmax = float(np.abs(v).max())                                  # the scale of each plate from its own rows only
        i, j = np.unravel_index(int(np.abs(v).argmax()), v.shape)
        fill[full] = dict(vmax=vmax, n=v.size, **{w: int((d == w).sum()) for w in ("win", "tie", "loss")})
        print(f"[world_model_grid] {'appendix' if full else 'main'}: {v.size} comparisons, W-T-L {fill[full]['win']}-"
              f"{fill[full]['tie']}-{fill[full]['loss']}; vmax = {vmax:.4f} ({BASELINES[rows[i]][1]} on {TASKS[j][1]}); "
              f"the largest loss {v.max():+.3f} is {v.max() / vmax:.1%} of vmax")
        save(draw(v, d, [counts[k] for k in rows], [BASELINES[k][1] for k in rows], vmax, full),
             ("world_model_grid_full" if full else "world_model_grid") + _SUFFIX)
    erb = BASELINES.index(("erb", "Neuron positions"))
    cap_main = CAPTION.format(erb=f"LeWi {tally(erb)}", **fill[False])
    cap_full = CAPTION_FULL.format(adopted=adopted, changes=changes, **fill[True])
    print("[world_model_grid] CAPTION:", cap_main)
    print("[world_model_grid] CAPTION_FULL:", cap_full)
    for stem, cap in (("world_model_grid", cap_main), ("world_model_grid_full", cap_full)):
        (OUT / f"{stem}{_SUFFIX}.caption.txt").write_text(cap + "\n")


if __name__ == "__main__":
    main()
