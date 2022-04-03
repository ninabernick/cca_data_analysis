#!/bin/bash

#SBATCH --job-name=processing_1
#SBATCH --time=10:00

source activate gemmrenv2
python postprocessing.py analysis_3_30/results
