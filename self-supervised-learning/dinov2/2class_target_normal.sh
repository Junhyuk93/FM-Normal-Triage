export PYTHONPATH="${PYTHONPATH}:." 
python  dinov2/run/eval/linear_1024.py \
    --config-file /workspace/weights/v2_stable_e400/t-n.yaml \
    --pretrained-weights /workspace/weights/v2_stable_e400/training_499999/teacher_checkpoint.pth \
    --output-dir /workspace/eval/v2_stable_e400/normal-triage-FM-2class_target_normal_balance_ratio \
    --train-dataset normal-triage:root=/workspace/dataset/v2/train \
    --val-dataset normal-triage:root=/workspace/dataset/v2/valid \
    --test-datasets normal-triage:root=/workspace/dataset/v2/valid