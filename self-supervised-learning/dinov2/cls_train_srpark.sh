export PYTHONPATH="${PYTHONPATH}:." 
python  dinov2/run/eval/linear.py \
    --config-file /workspace/weights/v2_stable_e400/config.yaml \
    --pretrained-weights /workspace/weights/v2_stable_e400/training_499999/teacher_checkpoint.pth \
    --output-dir /workspace/eval/v2_stable_e400/training_500000_new_sampler \
    --train-dataset osteo:root=/workspace/osteo_dataset/labelV6_AMC_FM_3cls/train \
    --val-dataset osteo:root=/workspace/osteo_dataset/labelV6_AMC_FM_3cls/valid \
    --test-datasets osteo:root=/workspace/osteo_dataset/labelV6_AMC_FM_3cls/test 

# /workspace/weights/teacher_checkpoint_125000.pth

# /workspace/weights/v2_stable_e200/training_124999/teacher_checkpoint.pth

 ### 기존 backbone 및 config
        # - /workspace/inference/weights_backbone/v1_backbone/teacher_checkpoint_125000.pth
        # - configs/ssl_our_config.yaml
        # - pretrained-linear: /workspace/inference/weights_linear/125000_512_31250_classifier_4_blocks_avgpool_True_lr_0_00100.pth
        ### 신규 v2 backbone
        # - v2_stable_e200
        #       - /workspace/weights/v2_stable_e200/training_124999/teacher_checkpoint.pth
        #       - /workspace/weights/v2_stable_e200/training_249999/teacher_checkpoint.pth
        # - v2_stable_e300
        #       - /workspace/weights/v2_stable_e300/training_187499/teacher_checkpoint.pth
        #       - /workspace/weights/v2_stable_e300/training_374999/teacher_checkpoint.pth
        # - v2_stable_e400
        #       - /workspace/weights/v2_stable_e400/training_249999/teacher_checkpoint.pth
        #       - /workspace/weights/v2_stable_e400/training_499999/teacher_checkpoint.pth
        # - v2_stable_e500
        #       - /workspace/weights/v2_stable_e500/training_312499/teacher_checkpoint.pth
        #       - /workspace/weights/v2_stable_e500/training_624999/teacher_checkpoint.pth
        # - v2_cropsize
        #       - /workspace/weights/v2_cropsize/training_124999/teacher_checkpoint.pth
        #       - /workspace/weights/v2_cropsize/training_249999/teacher_checkpoint.pth
        # - v2_normalize
        #       - /workspace/weights/v2_stable_normalize/training_249999/teacher_checkpoint.pth
        # - v2_no_colorjitter_imagenet
        #       - /workspace/weights/v2_stable_no_colorjitter_imagenet/training_249999/teacher_checkpoint.pth
        # - v2_no_colorjitter_znorm
        #       - /workspace/weights/v2_stable_no_colorjitter_znorm/training_249999/teacher_checkpoint.pth
        # - v2_challenge_1024_randerase (200 epoch trained)
        #       - /workspace/weights/v2_challenge_1024_randerase/training_249999/teacher_checkpoint.pth
        # - v2_stable_1024_e500
        #       - /workspace/weights/v2_stable_1024_e500/training_624999/teacher_checkpoint.pth
        # - v2_stable_1024_warmup_30
        #       - /workspace/weights/v2_stable_1024_warmup_30/training_624999/teacher_checkpoint.pth

        ### 신규 config:
        # - v2 stable e200~500, v2_stable_normalize
        #       - /workspace/inference/configs/v2_stable_config.yaml
        # - v2 cropsize 512 -> 1024
        #       - /workspace/weights/v2_cropsize/config.yaml
        # - v2_1024_randerase
        #       - /workspace/weights/v2_challenge_1024_randerase/config.yaml
        # - v2_stable_1024_e500, warmup_30
        #       - /workspace/weights/v2_stable_1024_e500/config.yaml
