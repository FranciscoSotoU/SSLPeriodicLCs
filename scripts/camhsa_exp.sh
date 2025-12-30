#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# Run all experiments sequentially
echo "Starting camhsa experiments..."

echo "Running metadata_lc experiment..."
python src/train.py experiment=MM/camhsa_exp_metadata_lc logger=wandb ++trainer.devices=[0] 

echo "Running metadata_meta experiment..." 
python src/train.py experiment=MM/camhsa_exp_metadata_meta logger=wandb ++trainer.devices=[0]

echo "Running metadata_both experiment..."
python src/train.py experiment=MM/camhsa_exp_metadata_both logger=wandb ++trainer.devices=[0] 

echo "All experiments completed."
