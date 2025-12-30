#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule_2.sh

python src/train.py experiment=multimodal_pretrained/mm_pt_simclr_long_exp.yaml
python src/train.py experiment=multimodal_pretrained/mm_pt_vicreg_long_exp.yaml