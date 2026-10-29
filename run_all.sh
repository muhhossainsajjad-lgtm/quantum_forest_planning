#!/usr/bin/env bash
# Reproduce all experiments, statistics and figures for the report.
set -e
cd "$(dirname "$0")/experiments"
python 01_single_forest_study.py
for part in A B C D; do python 02_benchmarks.py "$part"; done
python 03_analyze.py
python 04_diagrams.py
echo "Done. Results in results/data and results/figures."
