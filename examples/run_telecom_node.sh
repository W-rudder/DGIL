#!/bin/bash
python train_node.py \
  --data telecom_new \
  --config ./config/GIL.yml \
  --model ./models/Lorentz_no_fusion.pkl \
  --gpu 0 \
  --rand_node_features 50 \
  --patience 10 \
  --epoch 50 \
#  --posneg
