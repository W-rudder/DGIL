#!/bin/bash
python train.py \
  --data REDDIT \
  --config ./config/GIL.yml \
  --gpu 0 \
  --model_name rdt_no_fusion_nf \
  --rand_node_features 50 \
  # --rand_edge_features 20
