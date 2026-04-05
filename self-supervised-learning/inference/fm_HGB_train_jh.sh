python fm_HGB_train_jh.py \
  --config-file /workspace/inference/configs/v2_stable_config.yaml \
  --pretrained-weights /workspace/weights/v2_stable_e400/training_499999/teacher_checkpoint.pth \
  --pretrained-linear-list "/workspace/eval/v2_stable_e400/normal-triage-FM-v4/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v2/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v6/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v12/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v8/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v5/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v10/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v3/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v7/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v13/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v9/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v11/running_checkpoint_linear_eval_36250.pth" \
  --train-dataset normal-triage:root=/workspace/dataset/v3/train \
  --training-num-classes 3 \
  --clf-type logreg \
  --C 1.0 \
  --clf-out /workspace/inference/fm_HGB_weight.pickle

