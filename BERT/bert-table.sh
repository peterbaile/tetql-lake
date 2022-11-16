#!/bin/bash

# Loading the required module
source /etc/profile
conda deactivate
module load anaconda/2021b
conda activate

# Run the script
python table.py test