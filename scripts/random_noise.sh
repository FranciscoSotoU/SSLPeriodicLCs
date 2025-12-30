#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py experiment=vr_random_noise logger=wandb exp_name=vr_random_noise

python src/train.py experiment=myself_exp_fpv1_reduced_pretrained_linear_proj logger=wandb

