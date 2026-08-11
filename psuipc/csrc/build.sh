#!/usr/bin/env bash
# Build the C Gibbs sampler into psuipc/_csampler.dll.
#
# Set CC to select a compiler. The default is gcc.
# Run from the repository root or any other directory.
set -euo pipefail

CC="${CC:-gcc}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/sampler.c"
OUT="$HERE/../_csampler.dll"

# Static libgcc so the DLL depends only on system DLLs (KERNEL32, msvcrt) and loads
# from any shell without mingw on PATH -- needed because "c" is the default backend.
"$CC" -O3 -shared -static -static-libgcc -o "$OUT" "$SRC" -lm

echo "built $OUT"
