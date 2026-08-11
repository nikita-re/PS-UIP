"""
config.py
=========

Global configuration for the PS-UIP simulation study. All constants the
``psuipc`` package needs live here so the experiments are reproducible and easy
to tweak from a single place:

* data-generating constants (covariate structure, true outcome coefficients),
* the true treatment-effect values per endpoint,
* the propensity-score / borrowing tuning constants (Stage 1 and Stage 2),
* the superiority decision rule.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Covariate / design constants
# ---------------------------------------------------------------------------

# Number of baseline covariates X.
P_COVARIATES = 6

# AR(1) correlation for the covariate covariance matrix: Sigma_ij = AR1_RHO ** |i-j|.
AR1_RHO = 0.3

# ---------------------------------------------------------------------------
# True outcome-model coefficients (shared by the current RCT and the sources)
# ---------------------------------------------------------------------------

# Continuous endpoint.
CONT_BETA0 = 0.0
CONT_BETA = [0.5, 0.4, 0.3, 0.2, 0.2, 0.1]
CONT_SIGMA = 1.0

# Binary endpoint (log-odds scale). BIN_BETA0 sets the baseline event rate
# (~sigmoid(beta0)); -0.5 gives ~38% events, which keeps the per-subject Fisher
# information for the log-odds-ratio high so the conflict statistic can detect drift.
BIN_BETA0 = -0.5
BIN_BETA = [0.6, 0.5, 0.4, 0.3, 0.3, 0.2]

# ---------------------------------------------------------------------------
# True treatment-effect (theta) values per endpoint
# ---------------------------------------------------------------------------
#   continuous : additive treatment effect
#   binary     : conditional log odds ratio
THETA_TRUE = {
    "continuous": [0.0, 0.4],
    "binary": [0.0, math.log(1.8)],
}

# ---------------------------------------------------------------------------
# Stage 1: propensity-score design
# ---------------------------------------------------------------------------
PS_CLIP = (0.01, 0.99)    # propensity-score clipping bounds
MIN_RETAINED = 10         # minimum retained historical subjects in common support
MIN_ESS = 5.0             # minimum weighted (Kish) ESS for a usable source
N_OVERLAP_BINS = 20       # number of quantile bins for the overlap coefficient
WINSOR_PCT = 99.0         # winsorization percentile for the analysis weights

# ---------------------------------------------------------------------------
# Stage 2: prior-data-conflict discount
# ---------------------------------------------------------------------------
# For source k let
#   T_k = (muC_k - muC_cur)^2 / (SE_k^2 + SE_cur^2)
# be the standardized current-vs-historical CONTROL-mean discrepancy. Under
# exchangeability T_k ~ chi-square_1, so E[T_k] = 1 = one unit of evidence. The
# discount rho_k = min(1, exp(RHO_C - T_k)) = exp(-max(0, T_k - RHO_C)) borrows
# fully (rho_k = 1) while the discrepancy is within one chi-square unit, then decays
# EXPONENTIALLY in the excess conflict. RHO_C = 1 is the parameter-free "unit"
# threshold matching the unit-information scale of the prior. The same discount is
# applied per source and once to the overlap-weighted combined summary; each source
# is scaled by the stronger (smaller) of the two factors, so a drift shared across
# sources that is too mild to flag any single one is still caught.
RHO_C = 1.0

# PS-weighted power-prior borrow budget A (Lu 2022; Li 2025 competitor). Borrowed
# units per source a_k = min(A * r_k / sum_j r_j, m_k). None -> current control
# size n_C, so the power prior may borrow up to one control arm, allocated by overlap.
POWER_PRIOR_A = None

# Vague-component prior SD for the PS self-adapting mixture competitor (PS-SAM,
# Yang 2023; Zhao 2025). The SAM weight w = m_I/(m_I+m_V) mixes an informative prior
# N(mu_inf, 1/prec_inf) against a vague N(mu_inf, SAM_VAGUE_SD^2); a large value makes
# the vague component flat so w -> 1 under agreement and w -> 0 under control conflict.
SAM_VAGUE_SD = 10.0

# ---------------------------------------------------------------------------
# Analysis model and decision rule
# ---------------------------------------------------------------------------
WEAK_THETA_SD = 10.0      # weak prior on the treatment effect: theta ~ Normal(0, WEAK_THETA_SD^2)

# Superiority decision Pr(theta > 0 | data) > DECISION_CUT, a one-sided test at
# level alpha = 1 - DECISION_CUT. DECISION_CUT = 0.95 is the nominal one-sided 5%
# test; the type-I target in null scenarios and the figure reference line are 0.05.
DECISION_CUT = 0.95
