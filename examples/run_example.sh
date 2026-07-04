#!/usr/bin/env bash
set -euo pipefail

seed-hsi-passport \
  --input data/processed \
  --output results/optical_passport \
  --repeats 50 \
  --permutations 200 \
  --permutation-repeats 2 \
  --bootstrap 300 \
  --band-counts 1,3,5,10,20
