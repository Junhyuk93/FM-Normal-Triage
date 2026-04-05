export PYTHONPATH="${PYTHONPATH}:." 
python  dinov2/run/eval/linear_1024.py \
    --config-file /workspace/weights/v2_stable_e400/v5.yaml \
    --pretrained-weights /workspace/weights/v2_stable_e400/training_499999/teacher_checkpoint.pth \
    --output-dir /workspace/eval/v2_stable_e400/normal-triage-FM-v5 \
    --train-dataset normal-triage:root=/workspace/dataset/v5/train \
    --val-dataset normal-triage:root=/workspace/dataset/v5/valid \
    --test-datasets normal-triage:root=/workspace/dataset/v5/valid