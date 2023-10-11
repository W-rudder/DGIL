#!/bin/bash
python train_node.py \
  --data REDDIT \
  --config ./config/GIL.yml \
  --model ./models/rdt_no_fusion_nf.pkl \
  --gpu 0 \
  --patience 50 \
  --epoch 100 \
  --rand_node_features 50 \
#  --cheat \
#  --posneg
