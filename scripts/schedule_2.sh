#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule_2.sh

python src/train.py experiment=lightcurves/pt_atat_long_percentages
python src/train.py experiment=lightcurves/long_percentages