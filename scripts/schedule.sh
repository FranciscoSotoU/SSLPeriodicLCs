#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# Multimodal Classification Experiments
#python src/train.py experiment=multimodal/pt_long_percentages
# Lightcurve Only Classification Experiments
python src/train.py experiment=lightcurves/pt_long_percentages

# VICReg Linear Classifier Experiments
python src/train.py experiment=vicreg/pt_long
