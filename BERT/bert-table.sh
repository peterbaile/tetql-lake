#!/bin/bash

# Loading the required module
source /etc/profile
conda deactivate
module load anaconda/2021b
conda activate

# Run the script
# python table_ranking_join.py --mode train --path train_spider --join True
python table_ranking_join.py --mode dev --path train_spider --devfile dev_join --topk 4 --devpart 2 --join True