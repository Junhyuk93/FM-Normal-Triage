export PYTHONPATH="${PYTHONPATH}:." 
python  dinov2/run/eval/linear_1024.py \
    --config-file /workspace/weights/v2_stable_e400/v6.yaml \
    --pretrained-weights /workspace/weights/v2_stable_e400/training_499999/teacher_checkpoint.pth \
    --output-dir /workspace/eval/v2_stable_e400/normal-triage-FM-v6 \
    --train-dataset normal-triage:root=/workspace/dataset/v6/train \
    --val-dataset normal-triage:root=/workspace/dataset/v6/valid \
    --test-datasets normal-triage:root=/workspace/dataset/v6/valid