"""Recompute EWRL-2026 figures/tables for the SAC_ASYNC_TME_NEW_HYP_REWARD runs.

Train distribution = rectangle, test distribution = network-field (held-out).
Reads per-run CSVs downloaded by download_tme_new.py into wandb_tme_new/<mode>/.
Emits:
  - out/results_table.tex   (aggregate train/test mean+std per mode, seeds averaged)
  - out/train_return_mean50.pdf
  - out/test_return_mean50.pdf
  - out/return_std.pdf
Style conventions (palette, EWMA-50, mean+-1 std shading) follow plot_ewrl2026.py.
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

ACTION_REPEAT = 6  # each agent decision is held for 6 simulator steps

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "wandb_tme_new")
OUT = os.path.join(BASE, "out_tme_new")
os.makedirs(OUT, exist_ok=True)

# ── canonical short IDs + palette (curated main set flagged) ──────────────────
MODE_META = {
    "img_mc_cells_substrates_m1m2": dict(id="I2m", label="I2m img+subs+M1M2 (full info)",
                                          color="#b5179e", ls="-", lw=2.0, z=6, main=True),
    "img_mc_cells_substrates":      dict(id="I2", label="I2 img+substrates",
                                          color="#e63946", ls="-", lw=2.0, z=5, main=True),
    "img_mc_cells_m1m2":            dict(id="I1m", label="I1m img+M1M2",
                                          color="#f3722c", ls="-", lw=1.6, z=4, main=False),
    "img_mc_cells":                 dict(id="I1", label="I1 img cells only",
                                          color="#2a9d8f", ls="-", lw=1.8, z=4, main=True),
    "spatial_scalars_cells_substrates_m1m2": dict(id="S3sm", label="spatial scalars+subs+M1M2",
                                          color="#3a0ca3", ls="--", lw=1.5, z=3, main=False),
    "spatial_scalars_cells_substrates":      dict(id="S3s", label="spatial scalars+subs",
                                          color="#4361ee", ls="--", lw=1.5, z=3, main=True),
    "spatial_scalars_cells_m1m2":            dict(id="S3m", label="spatial scalars+M1M2",
                                          color="#4895ef", ls="--", lw=1.4, z=2, main=False),
    "spatial_scalars_cells_spatial_no_scalars_substrates_m1m2":
                                    dict(id="S5m", label="spatial scalars+spatial subs+M1M2",
                                          color="#57cc99", ls=":", lw=1.4, z=2, main=False),
    "scalars_macrophages":          dict(id="POMDP", label="scalars_macrophages (POMDP)",
                                          color="#adb5bd", ls="-.", lw=1.4, z=1, main=True),
}
MODE_ORDER = list(MODE_META.keys())


def _style():
    plt.rcParams.update({
        "font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def _ewma(a, w=50):
    a = np.asarray(a, float)
    if len(a) == 0:
        return a
    return uniform_filter1d(a, size=min(w, len(a)), mode="nearest")


import re as _re

MAX_SEEDS = 5  # fixed n=5 per mode; keep the 5 lowest seed-ids (first-5-by-seed-id)


def _seed_id(path):
    m = _re.search(r"seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 10**9


def load(mode, col):
    """Return list of (steps, values) per seed for a metric column.
    Capped at the MAX_SEEDS lowest seed-ids so every mode reports n=5."""
    out = []
    files = sorted(glob.glob(os.path.join(DATA, mode, "SAC_*.csv")), key=_seed_id)[:MAX_SEEDS]
    for f in files:
        df = pd.read_csv(f)[["step", col]].dropna()
        if len(df) > 3:
            out.append((df["step"].to_numpy(), df[col].to_numpy()))
    return out


def load_random(col):
    """Random-policy baseline seeds (obs mode is irrelevant for a random policy).
    Grouped under wandb_tme_new/random_baseline/."""
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, "random_baseline", "RANDOM_*.csv"))):
        df = pd.read_csv(f)[["step", col]].dropna()
        if len(df) > 3:
            out.append((df["step"].to_numpy(), df[col].to_numpy()))
    return out


def random_baseline_final():
    """Return dict with train/test mean+std for the random baseline (end of training)."""
    tr = [v[-50:].mean() for _, v in [(s, _ewma(v)) for s, v in load_random("train_return")]]
    te = [v[-20:].mean() for _, v in [(s, _ewma(v)) for s, v in load_random("test_return")]]
    tr, te = np.array(tr), np.array(te)
    return dict(n=len(tr), train_mu=tr.mean(), train_sd=tr.std(),
                test_mu=te.mean(), test_sd=te.std())


def resample(series, grid):
    """Interpolate each (steps,vals) onto common grid; return (n_seeds, len(grid))."""
    mat = []
    for steps, vals in series:
        vals = _ewma(vals)
        mat.append(np.interp(grid, steps, vals, left=np.nan, right=np.nan))
    return np.array(mat)


def aggregate_table():
    rows = []
    for mode in MODE_ORDER:
        m = MODE_META[mode]
        tr = [v[-50:].mean() for _, v in [(s, _ewma(v)) for s, v in load(mode, "train_return")]]
        te = [v[-20:].mean() for _, v in [(s, _ewma(v)) for s, v in load(mode, "test_return")]]
        tr, te = np.array(tr), np.array(te)
        rows.append(dict(id=m["id"], mode=mode, n=len(tr),
                         train_mu=tr.mean(), train_sd=tr.std(),
                         test_mu=te.mean(), test_sd=te.std(), main=m["main"]))
    rb = random_baseline_final()
    rows.append(dict(id="RAND", mode="random_baseline", n=rb["n"],
                     train_mu=rb["train_mu"], train_sd=rb["train_sd"],
                     test_mu=rb["test_mu"], test_sd=rb["test_sd"], main=True))
    return pd.DataFrame(rows)


def write_latex(df):
    def block(sub, title):
        s = []
        for _, r in sub.iterrows():
            s.append(f"    {r['id']} & \\texttt{{{r['mode'].replace('_',chr(92)+'_')}}} & {int(r['n'])} "
                     f"& {r['train_mu']:+.1f} & {r['train_sd']:.1f} "
                     f"& {r['test_mu']:+.1f} & {r['test_sd']:.1f} \\\\")
        return "\n".join(s)
    main = df[df.main].sort_values("test_mu", ascending=False)
    extra = df[~df.main].sort_values("test_mu", ascending=False)
    tex = f"""% auto-generated by plot_tme_new.py  (train=rectangle, test=network-field)
\\begin{{tabular}}{{llcccccc}}
  \\toprule
  \\textbf{{ID}} & \\textbf{{Mode}} & \\textbf{{$n$}} &
  \\textbf{{Train$_\\mu$}} & \\textbf{{Train$_\\sigma$}} &
  \\textbf{{Test$_\\mu$}}  & \\textbf{{Test$_\\sigma$}} \\\\
  \\midrule
{block(main,'main')}
  \\midrule
{block(extra,'extra')}
  \\bottomrule
\\end{{tabular}}
"""
    with open(os.path.join(OUT, "results_table.tex"), "w") as f:
        f.write(tex)
    df.to_csv(os.path.join(OUT, "results_table.csv"), index=False)


def curve_plot(col, ylabel, fname, main_only=True):
    _style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    gmax = 100000
    grid = np.linspace(0, gmax, 300)
    for mode in MODE_ORDER:
        m = MODE_META[mode]
        if main_only and not m["main"]:
            continue
        mat = resample(load(mode, col), grid)
        if mat.size == 0:
            continue
        mu = np.nanmean(mat, axis=0)
        sd = np.nanstd(mat, axis=0)
        x = grid * ACTION_REPEAT / 1e5
        ax.plot(x, mu, color=m["color"], ls=m["ls"], lw=m["lw"],
                zorder=m["z"], label=m["id"])
        ax.fill_between(x, mu - sd, mu + sd, color=m["color"], alpha=0.12, zorder=m["z"])
    # random-policy reference band (end-of-training mean +- std over seeds)
    rb = random_baseline_final()
    rb_mu = rb["train_mu"] if col == "train_return" else rb["test_mu"]
    rb_sd = rb["train_sd"] if col == "train_return" else rb["test_sd"]
    ax.axhline(rb_mu, color="k", lw=1.0, ls=(0, (4, 3)), alpha=0.7, zorder=0,
               label=f"random ({rb_mu:+.0f})")
    ax.axhspan(rb_mu - rb_sd, rb_mu + rb_sd, color="k", alpha=0.06, zorder=0)
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel(r"cumulative simulator steps ($\times 10^5$)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, ncol=2, loc="best")
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print("wrote", fname)


def std_plot(fname):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    grid = np.linspace(0, 100000, 300)
    for ax, col, ttl in [(axes[0], "train_return", "Train (rectangle)"),
                         (axes[1], "test_return", "Test (network-field)")]:
        for mode in MODE_ORDER:
            m = MODE_META[mode]
            if not m["main"]:
                continue
            mat = resample(load(mode, col), grid)
            if mat.size == 0:
                continue
            sd = np.nanstd(mat, axis=0)
            ax.plot(grid * ACTION_REPEAT / 1e5, sd, color=m["color"], ls=m["ls"], lw=m["lw"], label=m["id"])
        ax.set_title(ttl)
        ax.set_xlabel(r"simulator steps ($\times 10^5$)")
    axes[0].set_ylabel(r"return std across seeds")
    axes[0].legend(fontsize=8, ncol=2)
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print("wrote", fname)


if __name__ == "__main__":
    df = aggregate_table()
    print(df.to_string(index=False))
    write_latex(df)
    curve_plot("train_return", "training return (EWMA-50)", "train_return_mean50.pdf")
    curve_plot("test_return", "test return, network-field (EWMA-50)", "test_return_mean50.pdf")
    std_plot("return_std.pdf")
    print("\nAll outputs in", OUT)
