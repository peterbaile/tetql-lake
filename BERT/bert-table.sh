#!/bin/bash

# Loading the required module
source /etc/profile
conda deactivate
module load anaconda/2021b
conda activate

# Run the script
python table_ranking.py --mode dev --path train_spider --devfile dev_join --devpart 2 --addnegative true --topk 4