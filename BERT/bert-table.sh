#!/bin/bash

# Loading the required module
source /etc/profile
conda deactivate
module load anaconda/2021b
conda activate

# Run the script
# python table_ranking.py --mode train --path train_spider
python table_ranking_2.py --mode train --path train_spider

# python table_ranking_join.py --mode dev --path train_spider --devfile dev --topk 1 --devpart -1