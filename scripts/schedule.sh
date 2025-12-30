#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# Lightcurve Classification Experiments
python src/train.py experiment=lc/atat_periodic_vicreg_linear
python src/train.py experiment=lc/atat_periodic_vicreg