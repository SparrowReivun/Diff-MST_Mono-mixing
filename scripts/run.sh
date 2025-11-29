#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8        # 8 cores (8 cores per GPU)
#$ -l h_rt=1:0:0  # 240 hours runtime
#$ -l h_vmem=11G    # 11 * 8 = 88G total RAM
#$ -l gpu=1         # request 1 GPU
#$ -l gpu_type=ampere
 
source /data/home/eey818/Diff-MST/env/bin/activate
python /data/home/eey818/Diff-MST/scripts/eval_all_combo_Oct16.py