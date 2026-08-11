#!/usr/bin/env bash
# scripts/run_server.sh
# =====================
# Full-scale reproducibility workflow. Run from the repository root. The compiled
# C backend is fastest; use PSUIPC_BACKEND=fast if the C library is unavailable.
#
#   bash scripts/run_server.sh
#
# Simulation steps write psuipc/outputs/psuipc_{raw,summary}<tag>.csv and are
# independent. REPS/JOBS are environment-overridable for a quick check:
#   REPS=20 JOBS=4 bash scripts/run_server.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
export PYTHONPATH=.
export PSUIPC_BACKEND="${PSUIPC_BACKEND:-c}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
REPS="${REPS:-1000}"
JOBS="${JOBS:--1}"

echo "[run_server] backend=$PSUIPC_BACKEND reps=$REPS jobs=$JOBS"

# Main operating-characteristic grid, primary design.
"$PY" -m psuipc.run --methods main --reps "$REPS" --n-jobs "$JOBS" \
    --nC 100 --nT 100 --rh 3

# Second sample-size and historical-to-current ratio design.
"$PY" -m psuipc.run --methods main --reps "$REPS" --n-jobs "$JOBS" \
    --nC 60 --nT 120 --rh 5 --tag _nc60rh5

# Aggregate-discount ablation and matched PS-SAM comparison.
"$PY" -m psuipc.run --methods ablation --reps "$REPS" --n-jobs "$JOBS" \
    --nC 100 --nT 100 --rh 3 --tag _ablation

# Conflict-magnitude sweep and discount sensitivity analysis.
"$PY" -m psuipc.sweep --reps "$REPS" --n-jobs "$JOBS"
"$PY" -m psuipc.sensitivity --reps "$REPS" --n-jobs "$JOBS"

# ACTG175 real-data illustration.
"$PY" -m psuipc.application

echo "[run_server] done. Outputs in psuipc/outputs/."
