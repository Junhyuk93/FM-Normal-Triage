#!/bin/bash

export PYTHONPATH=..

python dinov2_infer.py \
    --config-file /workspace/inference/configs/v2_stable_config.yaml \
    --pretrained-weights /workspace/weights/v2_stable_e400/training_499999/teacher_checkpoint.pth \
    --test-dataset normal-triage:root=/workspace/dataset/vin_dataset \
    --batch-size 1 \
    --training-num-classes 3 \
    --pretrained-linear /workspace/eval/v2_stable_e400/normal-triage-FM/running_checkpoint_linear_eval_36250.pth
