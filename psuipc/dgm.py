"""
psuipc/dgm.py
=============

Data-generating mechanism for PS-UIP: a propensity-score-integrated unit
information prior that borrows ONLY historical CONTROL information into the
control arm of a current two-arm RCT (hybrid-control / augmented-control
design). This faithfully follows the augmented-control simulation conventions of
Wang, Suttner, Jemielita & Li (2022, JBS), Zhao, Yang, Laird, Chen & Yuan
(2025, JBS, PS-SAM), and the unit-information-prior DGM of Zhang & Yin (2023,
SMMR), but historical sources are CONTROL-ONLY.

Current RCT
-----------
    X ~ MVN(0, Sigma),  Z ~ Bernoulli(0.5)   (randomized, n_T treated / n_C control)
    continuous:  Y = beta0 + theta*Z + X beta + N(0, sigma^2)
    binary    :  logit P(Y=1) = beta0 + theta*Z + X beta

Historical control source k (NO treatment arm)
----------------------------------------------
    X_k ~ MVN(mu_k 1, Sigma)                 (covariate shift)
    continuous:  Y_k = beta0 + u_k + X_k beta + N(0, sigma^2)
    binary    :  logit P(Y_k=1) = beta0 + u_k + X_k beta

mu_k drives current-vs-historical covariate imbalance (membership-PS handles it);
u_k is an outcome drift / time trend on the control mean that the propensity
score CANNOT fix and that the UIP's borrowed information M must down-weight.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

import config

K_SOURCES = 3


def make_ar1_cov(p: int = None, rho: float = None) -> np.ndarray:
    """AR(1) covariance matrix Sigma with Sigma_ij = rho ** |i - j|."""
    p = config.P_COVARIATES if p is None else p
    rho = config.AR1_RHO if rho is None else rho
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

# Control-borrowing scenarios, grouped into three regimes of agreement between the
# historical and the current controls:
#   IDEAL      (I)      : no data-generating shift;
#   REALISTIC  (R1, R2) : moderate heterogeneity, the everyday external-control case
#                         where naive borrowing already starts to mislead but selective
#                         borrowing still extracts information;
#   STRESS     (S1..S3) : large conflict, where borrowing should collapse to no borrow.
# mu = covariate-shift mean, u = control-outcome drift (continuous additive / binary
# log-odds), nl = omitted outcome nonlinearity nl*(X1^2-1) (a functional-form
# misspecification the linear analysis model cannot represent), split = fraction of the
# historical pool in each source.
_EQUAL = [1 / 3, 1 / 3, 1 / 3]
SCENARIOS_CTRL = {
    # ---- Ideal ----
    "I": {"mu": [0.0, 0.0, 0.0],
          "u": {"continuous": [0.0, 0.0, 0.0], "binary": [0.0, 0.0, 0.0]},
          "split": _EQUAL, "regime": "Ideal", "desc": "No shift."},
    # ---- Realistic (moderate heterogeneity) ----
    # R1: moderate covariate shift with a mild outcome misspecification. Covariate
    # adjustment alone removes a pure linear shift, so the discriminating, realistic
    # case is misspecification: a non-transporting fit averages the omitted curvature
    # over the shifted region and is biased, while the PS fit transports to the current
    # region and stays robust, so it keeps borrowing where the others must not.
    "R1": {"mu": [0.4, 0.7, 1.0],
           "u": {"continuous": [0.0, 0.0, 0.0], "binary": [0.0, 0.0, 0.0]},
           "nl": [0.3, 0.3, 0.3],
           "split": _EQUAL, "regime": "Realistic",
           "desc": "Moderate covariate shift with outcome misspecification."},
    # R2: heterogeneous covariate shift with misspecification. The sources differ from
    # the current population to varying degrees, from mildly to strongly dissimilar,
    # with the same omitted nonlinearity. The membership PS transports each by its own
    # overlap, so borrowing is kept where it is safe and trimmed where it is not.
    "R2": {"mu": [0.3, 0.8, 1.3],
           "u": {"continuous": [0.0, 0.0, 0.0], "binary": [0.0, 0.0, 0.0]},
           "nl": [0.3, 0.3, 0.3],
           "split": _EQUAL, "regime": "Realistic",
           "desc": "Heterogeneous covariate shift with misspecification."},
    # ---- Stress (large conflict) ----
    # S1: a large shared positive control drift, as when the standard of care has
    # improved so the historical controls fare worse than the concurrent controls.
    "S1": {"mu": [0.0, 0.0, 0.0],
           "u": {"continuous": [0.4, 0.6, 0.9], "binary": [0.5, 0.8, 1.1]},
           "split": _EQUAL, "regime": "Stress",
           "desc": "Large positive outcome drift."},
    # S2: an adversarial, strongly dissimilar and lower-risk source that is also the
    # largest of the three, plus a misspecification.
    "S2": {"mu": [0.8, 1.2, 1.6],
           "u": {"continuous": [-0.4, -0.6, -0.9], "binary": [-0.5, -0.8, -1.1]},
           "nl": [0.3, 0.3, 0.3],
           "split": [0.2, 0.2, 0.6], "regime": "Stress",
           "desc": "Adversarial dissimilar lower-risk sources."},
    # S3: a large shared negative control drift, which biases the control mean down and
    # the treatment effect up, the direction that inflates the type-I error.
    "S3": {"mu": [0.0, 0.0, 0.0],
           "u": {"continuous": [-0.4, -0.6, -0.9], "binary": [-0.5, -0.8, -1.1]},
           "split": _EQUAL, "regime": "Stress",
           "desc": "Large negative outcome drift (type-I stress)."},
}
ALL_SCEN_CTRL = ["I", "R1", "R2", "S1", "S2", "S3"]


def _split_sizes(n_total: int, split: List[float]) -> List[int]:
    raw = [int(round(f * n_total)) for f in split]
    raw[-1] += n_total - sum(raw)
    return raw


def generate_current_rct(outcome, theta, nT, nC, rng) -> Dict:
    """Current two-arm RCT with randomized allocation (nT treated, nC control)."""
    p = config.P_COVARIATES
    Sig = make_ar1_cov()
    n = nT + nC
    X = rng.multivariate_normal(np.zeros(p), Sig, size=n)
    Z = np.concatenate([np.ones(nT), np.zeros(nC)])
    rng.shuffle(Z)
    if outcome == "continuous":
        b0, b = config.CONT_BETA0, np.asarray(config.CONT_BETA)
        Y = b0 + theta * Z + X @ b + rng.normal(0, config.CONT_SIGMA, n)
    else:
        b0, b = config.BIN_BETA0, np.asarray(config.BIN_BETA)
        Y = rng.binomial(1, _sigmoid(b0 + theta * Z + X @ b)).astype(float)
    return {"X": X, "Z": Z, "Y": Y, "nT": nT, "nC": nC,
            "X_C": X[Z == 0], "Y_C": Y[Z == 0],
            "X_T": X[Z == 1], "Y_T": Y[Z == 1]}


def generate_historical_control(outcome, n_k, mu_k, u_k, rng, nl_k=0.0) -> Dict:
    """One CONTROL-only historical source with covariate shift mu_k, outcome
    drift u_k, and an optional omitted nonlinearity nl_k. The nonlinear term
    nl_k * (X1^2 - 1) is centered (E[X1^2] = 1) so it does not act as a pure
    intercept shift; it is a functional-form misspecification that the linear
    analysis model cannot represent."""
    p = config.P_COVARIATES
    Sig = make_ar1_cov()
    X = rng.multivariate_normal(np.full(p, mu_k), Sig, size=n_k)
    g = nl_k * (X[:, 0] ** 2 - 1.0) if nl_k else 0.0
    if outcome == "continuous":
        b0, b = config.CONT_BETA0, np.asarray(config.CONT_BETA)
        Y = b0 + u_k + g + X @ b + rng.normal(0, config.CONT_SIGMA, n_k)
    else:
        b0, b = config.BIN_BETA0, np.asarray(config.BIN_BETA)
        Y = rng.binomial(1, _sigmoid(b0 + u_k + g + X @ b)).astype(float)
    return {"X": X, "Y": Y, "n": n_k}


def build_scenario_ctrl(scenario, outcome, theta, nT, nC, RH, rng,
                        u_override=None, mu_override=None, nl_override=None) -> Dict:
    """Build one current RCT + K historical control sources for a scenario. The
    optional u_override / mu_override / nl_override (length-K lists) replace the
    scenario's control-outcome drift / covariate-shift means / omitted nonlinearity,
    used by the conflict-magnitude sweep and scenario-tuning experiments."""
    spec = SCENARIOS_CTRL[scenario]
    nC_pool = RH * nC
    sizes = _split_sizes(nC_pool, spec["split"])
    nl = nl_override if nl_override is not None else spec.get("nl", [0.0] * K_SOURCES)
    mu = mu_override if mu_override is not None else spec["mu"]
    u = u_override if u_override is not None else spec["u"][outcome]
    cur = generate_current_rct(outcome, theta, nT, nC, rng)
    sources = []
    for k in range(K_SOURCES):
        sources.append(generate_historical_control(
            outcome, sizes[k], mu[k], u[k], rng, nl_k=nl[k]))
    return {"current": cur, "historical_controls": sources}
