"""
psuipc/sensitivity.py
=====================

Sensitivity of PS-UIP to the two modeling choices in the conflict discount: the
threshold c (anchored at one chi-square unit in the main analysis) and the decay
shape (exponential in the main analysis). For each setting the discount is refit on
the SAME simulated datasets as the main grid, so the differences isolate the tuning
choice. Reports type-I error, power, null RMSE and mean borrowed information for
PS-UIP under each setting, with no borrowing as the floor.

    python -m psuipc.sensitivity --reps 1000 --n-jobs -1

Settings: (c, shape) in {(0.5, exp), (1, exp)=default, (2, exp), (1, 1/T)}.
Output: psuipc/outputs/psuipc_sensitivity.csv.
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import math
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

THETA = {"continuous": [0.0, 0.4], "binary": [0.0, math.log(1.8)]}
# (label, threshold c, decay shape)
SETTINGS = [("c0.5", 0.5, "exp"), ("default", 1.0, "exp"),
            ("c2.0", 2.0, "exp"), ("invT", 1.0, "invT")]


def run_one(design, base_seed, idx, mcmc):
    outcome, scenario, theta, nT, nC, RH = design
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, idx]))
    inst = dgm.build_scenario_ctrl(scenario, outcome, theta, nT, nC, RH, rng)
    cur, srcs = inst["current"], inst["historical_controls"]
    rows = []
    # no borrowing floor (setting-independent).
    sd0 = int(np.random.SeedSequence([base_seed, idx, 0]).generate_state(1)[0])
    try:
        r0 = M.fit_no_borrowing(outcome, cur, None, mcmc, sd0)
    except Exception as e:
        r0 = {"failed": 1, "reason": str(e)}
    rows.append({"outcome": outcome, "scenario": scenario, "theta_true": theta,
                 "setting": "no_borrowing", **r0})
    # PS-UIP under each (c, shape) setting, on the same dataset.
    for si, (label, c, shape) in enumerate(SETTINGS):
        seed = int(np.random.SeedSequence([base_seed, idx, si + 1]).generate_state(1)[0])
        try:
            res = M.fit_ps_uip_c(outcome, cur, srcs, mcmc, seed, c=c, shape=shape)
        except Exception as e:
            res = {"failed": 1, "reason": f"{type(e).__name__}: {e}"}
        rows.append({"outcome": outcome, "scenario": scenario, "theta_true": theta,
                     "setting": label, **res})
    return rows


def aggregate(raw):
    out = []
    for (oc, sc, th, st), g in raw.groupby(["outcome", "scenario", "theta_true", "setting"]):
        ok = g[g["failed"].fillna(0).astype(int) == 0]
        R = len(ok)
        if R == 0:
            continue
        dec = ok["decision"].astype(float).mean()
        err = ok["theta_mean"].astype(float) - th
        out.append({
            "outcome": oc, "scenario": sc, "theta_true": round(th, 4), "setting": st,
            "R": R, "decision": dec, "decision_mcse": math.sqrt(max(dec * (1 - dec), 0) / R),
            "rmse": float(np.sqrt((err ** 2).mean())), "bias": float(err.mean()),
            "mean_M": float(ok["M_mean"].astype(float).mean()),
        })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--tune", type=int, default=300)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20260627)  # match run.py primary grid
    ap.add_argument("--nC", type=int, default=100)
    ap.add_argument("--nT", type=int, default=100)
    ap.add_argument("--rh", type=int, default=3)
    ap.add_argument("--outcomes", nargs="+", default=["continuous", "binary"])
    ap.add_argument("--scenarios", nargs="+", default=dgm.ALL_SCEN_CTRL)
    args = ap.parse_args()
    mcmc = {"draws": args.draws, "tune": args.tune, "chains": args.chains, "target_accept": 0.9}

    design = [(oc, sc, th, args.nT, args.nC, args.rh)
              for oc in args.outcomes for sc in args.scenarios for th in THETA[oc]]
    print(f"[sensitivity] backend={os.environ.get('PSUIPC_BACKEND','c')} "
          f"{len(design)} cells x {args.reps} reps x {len(SETTINGS)} settings")

    all_rows = Parallel(n_jobs=args.n_jobs, verbose=3)(
        delayed(run_one)(design[d], args.seed, d * args.reps + i, mcmc)
        for d in range(len(design)) for i in range(args.reps))
    raw = pd.DataFrame([r for sub in all_rows for r in sub])
    summ = aggregate(raw)
    summ.to_csv(os.path.join(OUTDIR, "psuipc_sensitivity.csv"), index=False)
    print(f"[sensitivity] wrote psuipc_sensitivity.csv ({len(summ)} rows)")

    t0 = summ[summ.theta_true == 0.0]
    for oc in args.outcomes:
        print(f"\n[{oc}] type-I by setting (theta=0)")
        print("  scen  " + "  ".join(f"{s[0]:>8}" for s in
                                      [("no_borrow",)] + SETTINGS))
        for sc in args.scenarios:
            def c(st):
                r = t0[(t0.outcome == oc) & (t0.scenario == sc) & (t0.setting == st)]
                return f"{r.iloc[0].decision:.3f}" if len(r) else "  -- "
            cells = "  ".join(f"{c(st):>8}" for st in
                              ["no_borrowing"] + [s[0] for s in SETTINGS])
            print(f"  {sc:>4}  {cells}")


if __name__ == "__main__":
    main()
