#!/bin/bash
python train_node.py \
  --data telecom_new \
  --config ./config/GIL.yml \
  --model ./models/Lorentz_no_edge.pkl \
  --gpu 0 \
  --rand_node_features 50 \
  --posneg