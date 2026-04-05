#!/usr/bin/env bash

export NCCL_DEBUG=INFO
export PYTHONPATH="${PYTHONPATH}:."
# export NCCL_IB_DISABLE=1

/home/workspace/miniconda3/envs/dinov2-h100/bin/python3.10 dinov2/run/train/train.py \
    --nodes 4 \
    --config-file /home/workspace/self-supervised-learning/dinov2/dinov2/configs/ssl_our_config.yaml \
    --output-dir ./results \
    train.dataset_path=PD:root=/home/dataset/FM_dataset

