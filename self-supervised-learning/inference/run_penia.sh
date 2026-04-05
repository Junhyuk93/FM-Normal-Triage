python inference.py --config-file /workspace/inference/configs/v2_stable_config.yaml \
        --pretrained-weights /workspace/weights/v2_stable_e400/training_499999/teacher_checkpoint.pth \
        --test-dataset penia:root=/workspace/dataset/3-cls-external/zzzcase_bohun  \
        --batch-size 1 \
        --training-num-classes 2 \
        --pretrained-linear /workspace/inference/weights_linear/penia/penia_32500_1K_classifier_4_blocks_avgpool_True_lr_0_00003.pth
        
        ##### best: /workspace/inference/weights_linear/v2_stable_e400_new/sampler_gelee_freeze_1K_30e_classifier_1_blocks_avgpool_True_lr_0_00125.pth

        ## Truncation(png) data
        # --test-dataset osteo:root=/workspace/osteo_dataset/zzzcase4/test \
        # --test-dataset osteo:root=/workspace/osteo_dataset/zzzcase4_cohort \
        # --test-dataset osteo:root=/workspace/osteo_dataset/zzzcase4_GH/
        # --test-dataset osteo:root=/workspace/osteo_dataset/GH_FM_3cls/test/
        # --test-dataset osteo:root=/workspace/osteo_dataset/labelV6_JBMR_FM_3cls/test
        # --test-dataset osteo:root=/workspace/osteo_dataset/labelV6_AMC_FM_3cls/test/
        # --test-dataset osteo:root=/workspace/Inference_datasets/zzzcase_bohun
        # --test-dataset osteo:root=/workspace/Inference_datasets/zzzcase_chuck

        ## labelv4: 
        # --test-dataset osteo:root=/workspace/osteo_dataset/zzzcase4_jbmr/ 
        # --test-dataset osteo:root=/workspace/osteo_dataset/zzzcase4/test \

        ## DCM format data
        # --test-dataset osteo:root=/workspace/osteo_dataset/labelV6_AMC_3cls_dcm/test
        # --test-dataset osteo:root=/workspace/osteo_dataset/cohort_1089_dcm
        # --test-dataset osteo:root=/workspace/jbmr_dcm
        # --test-dataset osteo:root=/workspace/Inference_datasets/chuck_dcm
        # --test-dataset osteo:root=/workspace/Inference_datasets/bohun_dcm
        # --test-dataset osteo:root=/workspace/osteo_dataset/gradienthealth-normal/PAs_clean
        # workspace/Inference_datasets/gh_all_dcm

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

        ### new trained classifier: default (512,512)
        # - v1_stable_e100_new: (v1 backbone으로 새로운 label v6 학습한 cls weight)
        #       - /workspace/inference/weights_linear/v1_stable_e100_new/10000_classifier_4_blocks_avgpool_False_lr_0_02500.pth
        #       - /workspace/inference/weights_linear/v1_stable_e100_new/sampler_7500_classifier_4_blocks_avgpool_False_lr_0_00250.pth
        # - v2_stable_e200_new:
        #       - 125000 iter: /workspace/inference/weights_linear/v2_stable_e200_new/21250_classifier_1_blocks_avgpool_True_lr_0_01000.pth
        #       - 250000 iter: /workspace/inference/weights_linear/v2_stable_e200_new/16250_classifier_4_blocks_avgpool_True_lr_0_00250.pth
        #                      /workspace/inference/weights_linear/v2_stable_e200_new/sampler_18750_classifier_4_blocks_avgpool_True_lr_0_00500.pth
        # - v2_stable_e300_new:
        #       - 187500 iter: /workspace/inference/weights_linear/v2_stable_e300_new/23750_classifier_1_blocks_avgpool_True_lr_0_01000.pth
        #       - 375000 iter: /workspace/inference/weights_linear/v2_stable_e300_new/10000_classifier_1_blocks_avgpool_True_lr_0_00500.pth
        #                      /workspace/inference/weights_linear/v2_stable_e300_new/sampler_18750_classifier_4_blocks_avgpool_False_lr_0_00100.pth
        # - v2_stable_e400_new:
        #       - 250000 iter: /workspace/inference/weights_linear/v2_stable_e400_new/17500_classifier_4_blocks_avgpool_False_lr_0_02500.pth
        #       - 500000 iter: /workspace/inference/weights_linear/v2_stable_e400_new/18750_classifier_1_blocks_avgpool_True_lr_0_00500.pth
        #                      /workspace/inference/weights_linear/v2_stable_e400_new_gelee_sampler/5000_classifier_1_blocks_avgpool_True_lr_0_00250.pth
        #                      /workspace/inference/weights_linear/v2_stable_e400_new/sampler_wo_layer_18750_classifier_4_blocks_avgpool_True_lr_0_00025.pth
        #                      /workspace/inference/weights_linear/v2_stable_e400_new/sampler_gelee_layer_7500_classifier_4_blocks_avgpool_True_lr_0_00001.pth
        #                      /workspace/inference/weights_linear/v2_stable_e400_new/sampler_gelee_layer_1K_37500_classifier_1_blocks_avgpool_True_lr_0_00025.pth
        #                      /workspace/inference/weights_linear/v2_stable_e400_new/sampler_gelee_no_trunc_16250_classifier_4_blocks_avgpool_True_lr_0_00250.pth
        #                      /workspace/inference/weights_linear/v2_stable_e400_new/sampler_gelee_unfreeze_e20_18750_classifier_4_blocks_avgpool_True_lr_0_00025.pth
        #                      /workspace/inference/weights_linear/v2_stable_e400_new/sampler_gelee_unfreeze_1K_e30_25000_classifier_4_blocks_avgpool_True_lr_0_00006.pth
        #                      /workspace/inference/weights_linear/v2_stable_e400_new/sampler_gelee_freeze_1K_30e_classifier_1_blocks_avgpool_True_lr_0_00125.pth
        #                      /workspace/inference/weights_linear/v2_stable_e400_new/sampler_gelee_freeze_1K_add_GH_classifier_4_blocks_avgpool_True_lr_0_00063.pth
        #                      /workspace/inference/weights_linear/v2_stable_e400_new/sampler_gelee_freeze_w_layer_13750_1K_classifier_1_blocks_avgpool_True_lr_0_00016.pth
        # - v2_stable_e500_new:
        #       - 312500 iter: /workspace/inference/weights_linear/v2_stable_e500_new/10000_classifier_1_blocks_avgpool_True_lr_0_02500.pth
        #       - 625000 iter: /workspace/inference/weights_linear/v2_stable_e500_new/6250_classifier_4_blocks_avgpool_True_lr_0_00100.pth
        #                      /workspace/inference/weights_linear/v2_stable_e500_new/sampler_5000_classifier_4_blocks_avgpool_True_lr_0_00050.pth
        # - v2_cropsize_new: (1024, 512)
        #       - 125000 iter: /workspace/inference/weights_linear/v2_cropsize_new/36250_classifier_4_blocks_avgpool_True_lr_0_00250.pth
        #       - 250000 iter: /workspace/inference/weights_linear/v2_cropsize_new/25000_classifier_1_blocks_avgpool_True_lr_0_00125.pth
        #                      /workspace/inference/weights_linear/v2_cropsize_new/sampler_50000_classifier_4_blocks_avgpool_True_lr_0_00125.pth
        # - v2_normalize_new:
        #       - 250000 iter: /workspace/inference/weights_linear/v2_normalize_new/13750_classifier_4_blocks_avgpool_False_lr_0_00500.pth
        #                      /workspace/inference/weights_linear/v2_normalize_new/sampler_5000_classifier_1_blocks_avgpool_True_lr_0_00500.pth
        # - v2_no_colorjitter_imagenet_new
        #       - 250000 iter: /workspace/inference/weights_linear/v2_no_colorjitter_imagenet_new/20000_classifier_1_blocks_avgpool_True_lr_0_01000.pth
        #                      /workspace/inference/weights_linear/v2_no_colorjitter_imagenet_new/sampler_18750_classifier_4_blocks_avgpool_True_lr_0_00500.pth
        # - v2_no_colorjitter_znorm_new
        #       - 250000 iter: /workspace/inference/weights_linear/v2_no_colorjitter_znorm_new/11250_classifier_4_blocks_avgpool_True_lr_0_01000.pth
        #                      /workspace/inference/weights_linear/v2_no_colorjitter_znorm_new/sampler_7500_classifier_4_blocks_avgpool_True_lr_0_00250.pth
        # - v2_challenge_1024_randerase_new
        #       - 2500000 iter: /workspace/inference/weights_linear/v2_challenge_1024_randerase_new/42500_classifier_4_blocks_avgpool_True_lr_0_00063.pth
        #                       /workspace/inference/weights_linear/v2_challenge_1024_randerase_new/sampler_41250_classifier_4_blocks_avgpool_True_lr_0_00013.pth
        # - v2_stable_1024_e500_new
        #       - 6250000 iter: /workspace/inference/weights_linear/v2_stable_1024_e500_new/60000_classifier_4_blocks_avgpool_True_lr_0_00013.pth
        #                       /workspace/inference/weights_linear/v2_stable_1024_e500_new/sampler_61250_classifier_1_blocks_avgpool_True_lr_0_00250.pth
        # - v2_stable_1024_warmup_30_new
        #       - 6250000 iter: /workspace/inference/weights_linear/v2_stable_1024_warmup_30_new/51250_classifier_1_blocks_avgpool_True_lr_0_00250
        #                       /workspace/inference/weights_linear/v2_stable_1024_warmup_30_new/sampler_56250_classifier_1_blocks_avgpool_True_lr_0_00250.pth
        # - v2_stable_e400_new_gelee_sampler
        #       - /workspace/inference/weights_linear/v2_stable_e400_new_gelee_sampler/5000_classifier_1_blocks_avgpool_True_lr_0_00250.pth
        # 
        #       - /workspace/inference/weights_linear/v2_stable_e400_new_phase2/phase2_best.pth
        #       - /workspace/inference/weights_linear/v2_stable_e400_new_phase2/phase2_elength.pth