#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2021b

# Run the script
# python table_ranking.py --mode train --path train_spider
# python table_ranking_2.py --mode train --path train_spider

# python ete_score.py --path train_spider --devfile dev_single --topk 1

# python ete_score.py --path train_spider --devfile dev_join_2 --topk 2 --rerank --topnum 15

python ete_score.py --devfile dev_em