"""
psuipc/sweep.py
===============

Conflict-magnitude sweep: the standard dynamic-borrowing dominance experiment
(cf. Yang et al. 2023 SAM; Jiang et al. 2023 elastic). Three sources with no covariate shift
historical control sources (mu = 0) all carry a COMMON control-outcome drift
u = delta; delta is swept from negative (historical controls lower than current,
which biases theta upward and inflates the type-I error of an over-borrower) through
zero (no conflict) to positive (biases theta downward and destroys an over-borrower's
power). At each delta the three borrowing priors are scored by

    type-I  = P(decision | theta = 0),
    RMSE0   = sqrt(mean (theta_hat - 0)^2 | theta = 0),
    power   = P(decision | theta = theta_alt).

The plot of these against delta is the efficient-frontier picture: PS-UIP holds the
type-I error near nominal and the RMSE near its no-conflict floor across the whole
range, while UIP and the PS power prior degrade on both as |delta| grows; at
delta = 0 all three coincide, so PS-UIP matches the full-borrow prior when there is
nothing to discount and never pays for its safety in the compatible case.

    python -m psuipc.sweep --reps 300 --n-jobs -1
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config
import psuipc.dgm as dgm
import psuipc.methods as M

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "outputs")
os.makedirs(OUTDIR, exist_ok=True)

METHODS = ["standard_uip", "ps_power_prior", "ps_uip_c"]
DELTAS = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
THETA = {"continuous": [0.0, 0.4], "binary": [0.0, float(np.log(1.8))]}


def run_one(outcome, delta, theta, base_seed, idx, mcmc):
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, idx]))
    inst = dgm.build_scenario_ctrl("I", outcome, theta, 100, 100, 3, rng,
                                   mu_override=[0.0, 0.0, 0.0],
                                   u_override=[delta, delta, delta])
    cur, srcs = inst["current"], inst["historical_controls"]
    rows = []
    for mi, m in enumerate(METHODS):
        seed = int(np.random.SeedSequence([base_seed, idx, mi]).generate_state(1)[0])
        try:
            res = M.METHODS[m](outcome, cur, srcs, mcmc, seed)
            dec, err = res["decision"], float(res["theta_mean"]) - theta
        except Exception:
            dec, err = np.nan, np.nan
        rows.append({"outcome": outcome, "delta": delta, "theta_true": theta,
                     "method": m, "decision": dec, "err": err})
    return rows


def aggregate(raw):
    out = []
    for (oc, d, th, m), g in raw.groupby(["outcome", "delta", "theta_true", "method"]):
        ok = g.dropna(subset=["decision"])
        R = len(ok)
        if R == 0:
            continue
        out.append({"outcome": oc, "delta": d, "theta_true": round(th, 4), "method": m,
                    "R": R, "decision": ok["decision"].mean(),
                    "rmse": float(np.sqrt((ok["err"] ** 2).mean())),
                    "bias": float(ok["err"].mean())})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--tune", type=int, default=400)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20260629)
    ap.add_argument("--outcomes", nargs="+", default=["continuous", "binary"])
    args = ap.parse_args()
    mcmc = {"draws": args.draws, "tune": args.tune, "chains": args.chains, "target_accept": 0.9}

    design = [(oc, d, th) for oc in args.outcomes for d in DELTAS for th in THETA[oc]]
    print(f"[sweep] {len(design)} cells x {args.reps} reps x {len(METHODS)} methods")
    rows = Parallel(n_jobs=args.n_jobs, verbose=3)(
        delayed(run_one)(design[c][0], design[c][1], design[c][2], args.seed,
                         c * args.reps + i, mcmc)
        for c in range(len(design)) for i in range(args.reps))
    raw = pd.DataFrame([r for sub in rows for r in sub])
    summ = aggregate(raw)
    summ.to_csv(os.path.join(OUTDIR, "psuipc_sweep.csv"), index=False)
    print(f"[sweep] wrote psuipc_sweep.csv ({len(summ)} rows)")


if __name__ == "__main__":
    main()
