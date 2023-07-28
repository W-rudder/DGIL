#!/bin/bash
python train.py \
  --data telecom_new \
  --config ./config/GIL.yml \
  --gpu -1 \
  --model_name Lorentz_rand_fusion \
  # --rand_node_features 50