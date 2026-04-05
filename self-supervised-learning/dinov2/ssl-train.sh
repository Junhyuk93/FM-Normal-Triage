#!/usr/bin/env bash

python dinov2/run/train/train.py \
    --nodes 4 \
    --ngpus 8 \
    --config-file /home/srpark/self-supervised-learning/dinov2/dinov2/configs/ssl_our_config.yaml \
    --output-dir ./results \
    train.dataset_path=PD:root=/home/dataset/FM_dataset

