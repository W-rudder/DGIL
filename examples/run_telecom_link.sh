#!/bin/bash
python train.py \
  --data Dgraph \
  --config ./config/GIL.yml \
  --gpu -1 \
  --model_name fin_no_fusion \
  # --rand_node_features 50 \
  # --rand_edge_features 20
