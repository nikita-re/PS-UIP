"""
psuipc/c_backend.py
===================

ctypes wrapper around psuipc/_csampler.dll (the C port of the numba Gibbs sampler
in fast_sampler.py). Exposes run_fixed / run_uip_fixed / run_uip with the SAME
signatures and SAME returned dict shape as fast_sampler.py, so methods.py can
dispatch to it via PSUIPC_BACKEND=c. It targets the identical posterior, so the
two backends are statistically consistent within Monte Carlo error.

The C function signature (see psuipc/csrc/sampler.c):

    void gibbs(const double* D, const double* y, int n, int q,
               int is_binary, int has_uip, int K,
               const double* muC, const double* I_U, const double* gamma,
               double M_lo, double M_hi,
               double beta0_mean0, double beta0_prec0,
               double weak_theta_var, double b_var,
               int n_draws, int n_tune, unsigned long long seed,
               double* theta_out, double* M_out, double* w_out);
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

# Reuse the design-matrix builder and the summary collector from the numba
# backend so the two stay byte-for-byte identical in those steps.
from psuipc.fast_sampler import _design, _collect

_HERE = os.path.dirname(os.path.abspath(__file__))
_DLL_PATH = os.path.join(_HERE, "_csampler.dll")

_lib = ctypes.CDLL(_DLL_PATH)

_dptr = ctypes.POINTER(ctypes.c_double)
_lib.gibbs.restype = None
_lib.gibbs.argtypes = [
    _dptr,                  # D
    _dptr,                  # y
    ctypes.c_int,           # n
    ctypes.c_int,           # q
    ctypes.c_int,           # is_binary
    ctypes.c_int,           # has_uip
    ctypes.c_int,           # K
    _dptr,                  # muC
    _dptr,                  # I_U
    _dptr,                  # gamma
    ctypes.c_double,        # M_lo
    ctypes.c_double,        # M_hi
    ctypes.c_double,        # beta0_mean0
    ctypes.c_double,        # beta0_prec0
    ctypes.c_double,        # weak_theta_var
    ctypes.c_double,        # b_var
    ctypes.c_int,           # n_draws
    ctypes.c_int,           # n_tune
    ctypes.c_ulonglong,     # seed
    _dptr,                  # theta_out
    _dptr,                  # M_out
    _dptr,                  # w_out
]


def _c(arr):
    """Coerce to a C-contiguous float64 array and return (array, pointer)."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    return a, a.ctypes.data_as(_dptr)


def _gibbs(D, y, is_binary, has_uip, K, muC, I_U, gamma, M_lo, M_hi,
           beta0_mean0, beta0_prec0, weak_theta_var, b_var,
           n_draws, n_tune, seed):
    """Thin wrapper calling the C gibbs(); returns (theta, M_out, w_mean)."""
    D, Dp = _c(D)
    y, yp = _c(y)
    n, q = int(D.shape[0]), int(D.shape[1])
    muC, muCp = _c(muC)
    I_U, I_Up = _c(I_U)
    gamma, gammap = _c(gamma)

    n_draws = int(n_draws)
    n_tune = int(n_tune)
    theta_out = np.empty(max(n_draws, 1), dtype=np.float64)
    M_out = np.empty(max(n_draws, 1), dtype=np.float64)
    w_out = np.zeros(int(K), dtype=np.float64)
    tp = theta_out.ctypes.data_as(_dptr)
    Mp = M_out.ctypes.data_as(_dptr)
    wp = w_out.ctypes.data_as(_dptr)

    _lib.gibbs(Dp, yp, n, q,
               int(bool(is_binary)), int(bool(has_uip)), int(K),
               muCp, I_Up, gammap,
               float(M_lo), float(M_hi),
               float(beta0_mean0), float(beta0_prec0),
               float(weak_theta_var), float(b_var),
               n_draws, n_tune, ctypes.c_ulonglong(int(seed) & 0xFFFFFFFFFFFFFFFF),
               tp, Mp, wp)

    return theta_out[:n_draws], M_out[:n_draws], w_out


# --------------------------------------------------------------------------- #
# Public wrappers mirroring fast_sampler.run_fixed / run_uip_fixed / run_uip.   #
# --------------------------------------------------------------------------- #
def run_fixed(outcome, X, Z, Y, center, beta0_var, weak_theta_var, mcmc, seed):
    """no_borrowing / pooling: fixed weak Normal(0, beta0_var) prior on beta0."""
    D = _design(X, Z, center)
    y = np.ascontiguousarray(np.asarray(Y, float))
    is_bin = outcome != "continuous"
    dummy = np.ones(1)
    theta, Md, wm = _gibbs(
        D, y, is_bin, False, 1, dummy, dummy, dummy, 1e-3, 1.0,
        0.0, 1.0 / beta0_var, weak_theta_var, 2.5 * 2.5,
        int(mcmc["draws"]) * int(mcmc.get("chains", 2)),
        int(mcmc["tune"]), int(seed))
    return _collect(theta, Md, wm, 1)


def run_uip_fixed(outcome, X, Z, Y, center, mu_prior, prec_prior,
                  weak_theta_var, mcmc, seed):
    """Deterministic UIP borrowing: fixed informative Normal(mu_prior, 1/prec_prior)
    prior on beta0 (standard_uip, ps_uip_c, ps_power_prior). M_mean is a placeholder
    the caller overwrites with the deterministic borrowed sample size."""
    D = _design(X, Z, center)
    y = np.ascontiguousarray(np.asarray(Y, float))
    is_bin = outcome != "continuous"
    dummy = np.ones(1)
    theta, Md, wm = _gibbs(
        D, y, is_bin, False, 1, dummy, dummy, dummy, 1e-3, 1.0,
        float(mu_prior), float(prec_prior), weak_theta_var, 2.5 * 2.5,
        int(mcmc["draws"]) * int(mcmc.get("chains", 2)),
        int(mcmc["tune"]), int(seed))
    return _collect(theta, Md, wm, 1)


def run_uip(outcome, X, Z, Y, center, muC, I_U, gamma, M_lo, M_hi,
            weak_theta_var, mcmc, seed):
    """UIP borrowing on beta0 with sampled (w, M). Returns (dict, w_mean)."""
    D = _design(X, Z, center)
    y = np.ascontiguousarray(np.asarray(Y, float))
    is_bin = outcome != "continuous"
    K = len(muC)
    theta, Md, wm = _gibbs(
        D, y, is_bin, True, K,
        np.ascontiguousarray(np.asarray(muC, float)),
        np.ascontiguousarray(np.asarray(I_U, float)),
        np.ascontiguousarray(np.asarray(gamma, float)),
        float(M_lo), float(M_hi), 0.0, 1.0, weak_theta_var, 2.5 * 2.5,
        int(mcmc["draws"]) * int(mcmc.get("chains", 2)),
        int(mcmc["tune"]), int(seed))
    return _collect(theta, Md, wm, K), wm
