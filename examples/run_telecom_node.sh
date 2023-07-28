#!/bin/bash
python train_node.py \
  --data telecom_new \
  --config ./config/GIL.yml \
  --model ./models/Lorentz_fusion.pkl \
  --gpu -1 \
  --rand_node_features 50 \
  --posneg