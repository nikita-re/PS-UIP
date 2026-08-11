"""
psuipc/application.py
=====================

Real-data application of PS-UIP on the ACTG175 trial (AIDS Clinical Trials Group
Study 175; Hammer et al. 1996), the open dataset distributed with the R package
speff2trial. ACTG175 is a single four-arm randomized trial, so we build an
augmented-control illustration from it in the standard hybrid-control way: a small
current two-arm trial, ZDV+ddI versus ZDV monotherapy, whose control arm is augmented
by three external control sources. Two are held-out ZDV-monotherapy controls (the
current control regimen), split by randomly halving the held-out pool, and one is the
didanosine-monotherapy arm (a different, conflicting regimen); the "mostly conflicting"
configuration reverses the majority to one monotherapy and two didanosine sources. The
endpoint is the CD4 count at week 20 (cd420), standardized to unit variance; the
estimand theta is the combination-versus-monotherapy effect.

This reproduces the numbers in the Application section. Run from the repository root:

    python -m psuipc.application
"""

from __future__ import annotations

import os
import urllib.request

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PSUIPC_BACKEND", "fast")

import warnings

import numpy as np
import pandas as pd

import psuipc.methods as M

warnings.filterwarnings("ignore")

DATA_URL = "https://raw.githubusercontent.com/cran/speff2trial/master/data/ACTG175.txt"
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "ACTG175.txt")
COVS = ["age", "wtkg", "karnof", "cd40", "cd80", "symptom"]   # p = 6 covariates
SEED = 20260701
NC_CUR, NT_CUR = 50, 100        # current-trial control and treatment sizes
SRC_CAP = 150                   # cap per external source


def _fetch():
    """Download ACTG175 robustly (full read), validating the row count."""
    with urllib.request.urlopen(DATA_URL, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    if text.count("\n") < 2000:
        raise RuntimeError("ACTG175 download looks truncated; retry")
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        f.write(text)


def load():
    if not os.path.exists(DATA_PATH) or sum(1 for _ in open(DATA_PATH)) < 2000:
        _fetch()
    df = pd.read_csv(DATA_PATH, sep=r"\s+").dropna(
        subset=COVS + ["cd420", "arms", "strat"]).copy()
    ymu, ysd = df["cd420"].mean(), df["cd420"].std()
    df["Yz"] = (df["cd420"] - ymu) / ysd
    for c in COVS[:5]:          # standardize the continuous covariates
        df[c] = (df[c] - df[c].mean()) / df[c].std()
    return df, ymu, ysd


def _xy(df, idx):
    d = df.loc[idx]
    return d[COVS].to_numpy(float), d["Yz"].to_numpy(float)


def gold_standard(df):
    """Reference effect estimated from the FULL trial: all ZDV-monotherapy controls
    versus all ZDV+ddI treated, covariate-adjusted. This is a correlated reference,
    not a known treatment effect."""
    import statsmodels.api as sm
    d = df[df.arms.isin([0, 1])].copy()
    d["Z"] = (d.arms == 1).astype(float)
    X = sm.add_constant(d[["Z"] + COVS].to_numpy(float))
    return float(sm.OLS(d["Yz"].to_numpy(float), X).fit().params[1])


def build(df, rng=None, adversarial=False):
    """Build the current trial and three external control sources.

    The default ("friendly") configuration uses two held-out ZDV-monotherapy sources
    (the current control regimen; compatible) and one ddI-monotherapy source (a
    different regimen; conflicting). The "adversarial" configuration reverses the
    majority, one compatible ZDV-monotherapy source and two conflicting ddI-monotherapy
    sources, to test the method where most external controls are in conflict.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    ctrl = df[df.arms == 0]     # ZDV monotherapy = control regimen
    trt = df[df.arms == 1]      # ZDV + ddI = treatment regimen
    ddi = df[df.arms == 3]      # ddI monotherapy = a different control regimen (conflict)
    cur_c = rng.choice(ctrl.index, size=NC_CUR, replace=False)
    cur_t = rng.choice(trt.index, size=NT_CUR, replace=False)
    ext = ctrl.drop(index=cur_c)

    Xc, Yc = _xy(df, cur_c)
    Xt, Yt = _xy(df, cur_t)
    cur = {"X": np.vstack([Xt, Xc]), "Z": np.r_[np.ones(NT_CUR), np.zeros(NC_CUR)],
           "Y": np.r_[Yt, Yc], "X_C": Xc, "Y_C": Yc, "X_T": Xt, "Y_T": Yt,
           "nT": NT_CUR, "nC": NC_CUR}

    def take(idx):
        idx = np.asarray(idx)
        if len(idx) > SRC_CAP:
            idx = rng.choice(idx, size=SRC_CAP, replace=False)
        Xs, Ys = _xy(df, idx)
        return {"X": Xs, "Y": Ys, "n": len(idx)}

    zdv = rng.permutation(ext.index.to_numpy())
    ddi_idx = rng.permutation(ddi.index.to_numpy())
    if not adversarial:
        h = len(zdv) // 2
        srcs = [take(zdv[:h]), take(zdv[h:]), take(ddi_idx)]
        labels = ("ZDV-mono A", "ZDV-mono B", "ddI-mono")
    else:
        h = len(ddi_idx) // 2
        srcs = [take(zdv), take(ddi_idx[:h]), take(ddi_idx[h:])]
        labels = ("ZDV-mono", "ddI-mono A", "ddI-mono B")
    return cur, srcs, labels


METHODS_ORDER = [("No borrowing", "no_borrowing"), ("UIP", "standard_uip"),
                 ("PS-PP", "ps_power_prior"), ("PS-UIP", "ps_uip_c")]
MCOLOR = {"No borrowing": "#777777", "UIP": "#1f77b4",
          "PS-PP": "#ff7f0e", "PS-UIP": "#2ca02c"}
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
FIG_PATH = os.path.join(OUTPUT_DIR, "psuipc_application.png")


def make_figure(ms_f, ms_a, rep_p, rep_labels):
    """Figure from the 30-split summaries ms_f (friendly) and ms_a (adversarial),
    each a dict method -> (bias_mean, bias_sd, width_mean, width_sd), plus the
    per-source diagnostics rep_p of one representative friendly split."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labs = [l for l, _ in METHODS_ORDER]
    keys = [k for _, k in METHODS_ORDER]
    x = np.arange(len(labs))
    w = 0.38
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))

    # (a) absolute deviation from the full-trial reference over partitions
    bf = [ms_f[k][0] for k in keys]; bfsd = [ms_f[k][1] for k in keys]
    ba = [ms_a[k][0] for k in keys]; basd = [ms_a[k][1] for k in keys]
    ax[0].bar(x - w / 2, bf, w, yerr=bfsd, capsize=3, color="#9ecae1",
              label="mostly compatible")
    ax[0].bar(x + w / 2, ba, w, yerr=basd, capsize=3, color="#fc9272",
              label="mostly conflicting")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labs, fontsize=8)
    ax[0].set_ylabel("$|\\hat\\theta - \\theta^*|$")
    ax[0].set_title("(a) Absolute deviation from full-trial reference")
    ax[0].legend(fontsize=7)

    # (b) credible-interval width, by configuration
    wf = [ms_f[k][2] for k in keys]; wfsd = [ms_f[k][3] for k in keys]
    wa = [ms_a[k][2] for k in keys]; wasd = [ms_a[k][3] for k in keys]
    ax[1].bar(x - w / 2, wf, w, yerr=wfsd, capsize=3, color="#9ecae1",
              label="mostly compatible")
    ax[1].bar(x + w / 2, wa, w, yerr=wasd, capsize=3, color="#fc9272",
              label="mostly conflicting")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labs, fontsize=8)
    ax[1].set_ylabel("$95\\%$ CrI width")
    ax[1].set_title("(b) Interval width")
    ax[1].legend(fontsize=7)

    # (c) per-source borrowing on one representative split
    units = [rep_p.get("m%d" % (i + 1)) * rep_p.get("r%d" % (i + 1))
             * rep_p.get("rho%d" % (i + 1)) for i in range(3)]
    rho = [rep_p.get("rho%d" % (i + 1)) for i in range(3)]
    cols = ["#2ca02c" if rk >= 0.5 else "#d62728" for rk in rho]
    xb = np.arange(3)
    ax[2].bar(xb, units, color=cols)
    for xi, u, rk in zip(xb, units, rho):
        ax[2].text(xi, u + 1, f"$\\rho^\\star$={rk:.2f}", ha="center", fontsize=8)
    ax[2].set_xticks(xb); ax[2].set_xticklabels(rep_labels, fontsize=8)
    ax[2].set_ylabel("borrowed units $m_k r_k \\rho_k^\\star$")
    ax[2].set_title("(c) Per-source borrowing, one split")

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print("wrote", FIG_PATH)


def run():
    df, ymu, ysd = load()
    cur, sources, src_labels = build(df)
    theta_star = gold_standard(df)
    mcmc = dict(draws=1000, tune=1000, chains=2, target_accept=0.9)
    res = {}
    for lab, key in METHODS_ORDER:
        res[key] = M.METHODS[key]("continuous", cur, sources, mcmc, 7)
    return df, cur, sources, src_labels, theta_star, ymu, ysd, res


def multisplit(df, theta_star, n_splits=30, adversarial=False):
    """Repeat the augmented-control analysis over many random splits and return, per
    method, the mean and SD of the absolute deviation from the full-trial reference
    and of the CrI width."""
    mcmc = dict(draws=800, tune=800, chains=2, target_accept=0.9)
    bias = {k: [] for _, k in METHODS_ORDER}
    width = {k: [] for _, k in METHODS_ORDER}
    for s in range(n_splits):
        rng = np.random.default_rng(1000 + s)
        cur, sources, _ = build(df, rng=rng, adversarial=adversarial)
        for _, key in METHODS_ORDER:
            r = M.METHODS[key]("continuous", cur, sources, mcmc, 100 + s)
            bias[key].append(abs(r["theta_mean"] - theta_star))
            width[key].append(r["ci_hi"] - r["ci_lo"])
    out = {}
    for _, k in METHODS_ORDER:
        out[k] = (np.mean(bias[k]), np.std(bias[k]), np.mean(width[k]), np.std(width[k]))
    return out


def main():
    df, cur, sources, src_labels, theta_star, ymu, ysd, res = run()
    print(f"ACTG175 cd420 (mean {ymu:.0f}, sd {ysd:.0f} cells); current nT={NT_CUR}, "
          f"nC={NC_CUR}; external {[s['n'] for s in sources]} {list(src_labels)}")
    print(f"full-trial reference theta* = {theta_star:+.3f} sd\n")
    print(f"{'method':<14}{'theta':>8}{'|dev|':>8}{'95% CrI':>20}{'width':>8}{'M':>7}")
    for lab, key in METHODS_ORDER:
        r = res[key]
        print(f"{lab:<14}{r['theta_mean']:>+8.3f}{abs(r['theta_mean']-theta_star):>8.3f}"
              f"{'[%+.3f, %+.3f]' % (r['ci_lo'], r['ci_hi']):>20}"
              f"{r['ci_hi']-r['ci_lo']:>8.3f}{r['M_mean']:>7.0f}")
    p = res["ps_uip_c"]
    print("\nPS-UIP per-source (label: m, r, rho):")
    for i, lab in enumerate(src_labels):
        print(f"  {lab:<16} m={p.get('m%d'%(i+1)):.0f}  r={p.get('r%d'%(i+1)):.2f}  "
              f"rho={p.get('rho%d'%(i+1)):.2f}")
    summaries = {}
    for adv, name in [(False, "friendly (2 compatible, 1 conflicting)"),
                      (True, "adversarial (1 compatible, 2 conflicting)")]:
        ms = multisplit(df, theta_star, n_splits=30, adversarial=adv)
        summaries[adv] = ms
        print(f"\n30-split summary, {name}  (mean+/-SD of |deviation| and CrI width):")
        print(f"  {'method':<13}{'|deviation|':>16}{'width':>16}")
        for lab, key in METHODS_ORDER:
            b, bs, w, ws = ms[key]
            print(f"  {lab:<13}{f'{b:.3f}+/-{bs:.3f}':>16}{f'{w:.3f}+/-{ws:.3f}':>16}")
    make_figure(summaries[False], summaries[True], p, src_labels)


if __name__ == "__main__":
    main()
