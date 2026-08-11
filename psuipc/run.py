"""
psuipc/run.py
=============

Operating-characteristic Monte Carlo for PS-UIP with Monte Carlo standard
errors (MCSE). Decision rule Pr(theta>0|data) > 0.95. Borrows historical
CONTROL information only.

    python -m psuipc.run --reps 60 --n-jobs -1
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

# Method presets. "main" is the paper's headline suite; "ablation" adds the
# revision experiments (#2 rho_g ablation, #5 PS-SAM competitor); "all" is the union.
METHOD_SETS = {
    "main": ["no_borrowing", "pooling", "standard_uip", "ps_power_prior", "ps_uip_c"],
    "ablation": ["no_borrowing", "standard_uip", "ps_power_prior", "ps_uip_c",
                 "ps_uip_psonly", "ps_sam"],
    "all": ["no_borrowing", "pooling", "standard_uip", "ps_power_prior", "ps_uip_c",
            "ps_uip_psonly", "ps_sam"],
}
ALL_FITS = {**M.METHODS, **M.METHODS_ABLATION}
THETA = {"continuous": [0.0, 0.4], "binary": [0.0, math.log(1.8)]}


def run_one(design, base_seed, idx, mcmc, method_order):
    outcome, scenario, theta, nT, nC, RH = design
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, idx]))
    inst = dgm.build_scenario_ctrl(scenario, outcome, theta, nT, nC, RH, rng)
    cur, srcs = inst["current"], inst["historical_controls"]
    rows = []
    for mi, m in enumerate(method_order):
        seed = int(np.random.SeedSequence([base_seed, idx, mi]).generate_state(1)[0])
        try:
            res = ALL_FITS[m](outcome, cur, srcs, mcmc, seed)
        except Exception as e:
            res = {"method": m, "failed": 1, "reason": f"{type(e).__name__}: {e}"}
        row = {"outcome": outcome, "scenario": scenario, "theta_true": theta}
        row.update(res)
        rows.append(row)
    return rows


def aggregate(raw):
    out = []
    for (oc, sc, th, m), g in raw.groupby(["outcome", "scenario", "theta_true", "method"]):
        ok = g[g["failed"].fillna(0).astype(int) == 0]
        R = len(ok)
        if R == 0:
            continue
        dec = ok["decision"].astype(float).mean()
        err = ok["theta_mean"].astype(float) - th
        cov = ((ok["ci_lo"] <= th) & (th <= ok["ci_hi"])).astype(float).mean()
        out.append({
            "outcome": oc, "scenario": sc, "theta_true": round(th, 4), "method": m, "R": R,
            "decision": dec, "decision_mcse": math.sqrt(max(dec * (1 - dec), 0) / R),
            "coverage": cov, "coverage_mcse": math.sqrt(max(cov * (1 - cov), 0) / R),
            "bias": float(err.mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "mean_M": float(ok["M_mean"].astype(float).mean()),
            "mean_rhat": float(ok["rhat"].astype(float).mean()),
        })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--tune", type=int, default=300)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20260627)
    ap.add_argument("--nC", type=int, default=100)
    ap.add_argument("--nT", type=int, default=100)
    ap.add_argument("--rh", type=int, default=3)
    ap.add_argument("--outcomes", nargs="+", default=["continuous", "binary"])
    ap.add_argument("--scenarios", nargs="+", default=dgm.ALL_SCEN_CTRL)
    ap.add_argument("--methods", default="main", choices=list(METHOD_SETS),
                    help="method preset: main (paper suite), ablation (+rho_g ablation "
                         "and PS-SAM), or all")
    ap.add_argument("--tag", default="", help="suffix for the output CSV names, e.g. "
                    "_nc60 for a second sample-size cell")
    args = ap.parse_args()
    method_order = METHOD_SETS[args.methods]
    mcmc = {"draws": args.draws, "tune": args.tune, "chains": args.chains, "target_accept": 0.9}

    design = []
    for oc in args.outcomes:
        for sc in args.scenarios:
            for th in THETA[oc]:
                design.append((oc, sc, th, args.nT, args.nC, args.rh))
    print(f"[psuipc] backend={os.environ.get('PSUIPC_BACKEND','c')} "
          f"methods={args.methods} ({len(method_order)}) "
          f"{len(design)} cells x {args.reps} reps")

    all_rows = Parallel(n_jobs=args.n_jobs, verbose=3)(
        delayed(run_one)(design[d], args.seed, d * args.reps + i, mcmc, method_order)
        for d in range(len(design)) for i in range(args.reps))
    raw = pd.DataFrame([r for sub in all_rows for r in sub])
    raw.to_csv(os.path.join(OUTDIR, f"psuipc_raw{args.tag}.csv"), index=False)
    summ = aggregate(raw)
    summ.to_csv(os.path.join(OUTDIR, f"psuipc_summary{args.tag}.csv"), index=False)
    print(f"[psuipc] wrote summary{args.tag} ({len(summ)} rows)")

    t0 = summ[summ.theta_true == 0.0]
    for oc in args.outcomes:
        print(f"\n[{oc}] type-I (theta=0)  scen  no_b  pool  stdUIP  PSpow  PS-UIP  | PS-UIP cov  meanM")
        for sc in args.scenarios:
            def c(m, col="decision"):
                r = t0[(t0.outcome == oc) & (t0.scenario == sc) & (t0.method == m)]
                return f"{r.iloc[0][col]:.3f}" if len(r) else "  -- "
            print(f"                     {sc:>4}  {c('no_borrowing')} {c('pooling')} "
                  f"{c('standard_uip')} {c('ps_power_prior')} {c('ps_uip_c')}     "
                  f"{c('ps_uip_c','coverage')}   {c('ps_uip_c','mean_M')}")


if __name__ == "__main__":
    main()
