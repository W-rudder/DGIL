#!/bin/bash
python train.py \
  --data telecom_new \
  --config ./config/GIL.yml \
  --gpu 0 \
  --model_name Lorentz_neg \
  --rand_node_features 50 \
  # --rand_edge_features 20