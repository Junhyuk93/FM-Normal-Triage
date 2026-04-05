#!/bin/bash
export PYTHONPATH=..
export XFORMERS_DISABLED=1

# SVM 모델 경로 설정
SVM_MODEL_PATH="/workspace/inference/svm_weight.pickle"

# 13개 FM 모델 + SVM inference 실행
python fm_svm_infer2.py \
  --config-file /workspace/inference/configs/v2_stable_config.yaml \
  --pretrained-weights /workspace/weights/v2_stable_e400/training_499999/teacher_checkpoint.pth \
  --pretrained-linear-list "/workspace/eval/v2_stable_e400/normal-triage-FM-v4/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v2/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v6/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v12/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v8/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v5/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v10/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v3/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v7/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v13/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v9/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v11/running_checkpoint_linear_eval_36250.pth" \
  --test-dataset normal-triage:root=/workspace/dataset/v3/valid \
  --batch-size 1 \
  --training-num-classes 3 \
  --svm-model-path "$SVM_MODEL_PATH"

echo "13-Model FM + SVM inference completed!"
