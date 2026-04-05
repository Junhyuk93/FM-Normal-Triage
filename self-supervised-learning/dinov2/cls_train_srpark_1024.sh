export PYTHONPATH="${PYTHONPATH}:." 
python  dinov2/run/eval/linear_1024.py \
    --config-file /workspace/weights/v2_stable_e400/config_to_1K.yaml \
    --pretrained-weights /workspace/weights/v2_stable_e400/training_499999/teacher_checkpoint.pth \
    --output-dir /workspace/eval/v2_stable_e400/normal-triage-FM \
    --train-dataset normal-triage:root=/workspace/dataset/v1/train \
    --val-dataset normal-triage:root=/workspace/dataset/v1/valid \
    --test-datasets normal-triage:root=/workspace/dataset/v1/valid

    # Truncation image
    # --train-dataset osteo:root=/workspace/osteo_dataset/labelV6_AMC_FM_3cls/train \
    # --val-dataset osteo:root=/workspace/osteo_dataset/labelV6_AMC_FM_3cls/valid \
    # --test-datasets osteo:root=/workspace/osteo_dataset/labelV6_AMC_FM_3cls/test 

    # Truncation Image + GH(new; Truncated)
    # --train-dataset osteo:root=/workspace/Asan_GH_Downstream/train \
    # --val-dataset osteo:root=/workspace/Asan_GH_Downstream/valid \
    # --test-datasets osteo:root=/workspace/Asan_GH_Downstream/test

    # DCM image (no truncation)
    # --train-dataset osteo:root=/workspace/osteo_dataset/labelV6_AMC_3cls_dcm/train \
    # --val-dataset osteo:root=/workspace/osteo_dataset/labelV6_AMC_3cls_dcm/valid \
    # --test-datasets osteo:root=/workspace/osteo_dataset/labelV6_AMC_3cls_dcm/test 

# - v2_cropsize
#       - /workspace/weights/v2_cropsize/training_124999/teacher_checkpoint.pth
#       - /workspace/weights/v2_cropsize/training_249999/teacher_checkpoint.pth
# - v2_challenge_1024_randerase (200 epoch trained)
#       - /workspace/weights/v2_challenge_1024_randerase/training_249999/teacher_checkpoint.pth
# - v2_stable_1024_e500
#       - /workspace/weights/v2_stable_1024_e500/training_624999/teacher_checkpoint.pth
# - v2_stable_1024_warmup_30
#       - /workspace/weights/v2_stable_1024_warmup_30/training_624999/teacher_checkpoint.pth
# - v2_stable_e400 (512 -> 1K)
#       - /workspace/weights/v2_stable_e400/training_499999/teacher_checkpoint.pth

# - v2 cropsize 512 -> 1024
#       - /workspace/weights/v2_cropsize/config.yaml
# - v2_1024_randerase
#       - /workspace/weights/v2_challenge_1024_randerase/config.yaml
# - v2_stable_1024_e500, warmup_30
#       - /workspace/weights/v2_stable_1024_e500/config.yaml
#       - /workspace/weights/v2_stable_1024_warmup_30/config.yaml
# - v2_stable_e400 (512 -> 1K)
#       - /workspace/weights/v2_stable_e400/config_to_1K.yaml (<- 기존꺼랑 다르지 않음!)