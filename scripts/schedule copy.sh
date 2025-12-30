#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# Lightcurve Classification Experiments

# Baseline ATAT Experiment
#python src/train.py experiment=lc/atat_periodic 

# DiT Architecture Experiments
#python src/train.py experiment=lc/dit_periodic 
#python src/train.py experiment=lc/dit_periodic_simclr 
#python src/train.py experiment=lc/dit_periodic_vicreg

# DiViT Architecture Experiments
#python src/train.py experiment=lc/divit_periodic
#python src/train.py experiment=lc/divit_periodic_vicreg

# DiViT Large Architecture Experiments
#python src/train.py experiment=lc/divit_L_periodic
#python src/train.py experiment=lc/divit_L_periodic_vicreg

# Linear Classification Experiments (Frozen Encoders)
python src/train.py experiment=lc/dit_periodic_vicreg_linear 
python src/train.py experiment=lc/dit_periodic_simclr_linear
#python src/train.py experiment=lc/divit_periodic_vicreg_linear
#python src/train.py experiment=lc/divit_L_periodic_vicreg_linear


