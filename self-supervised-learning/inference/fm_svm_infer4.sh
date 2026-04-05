python fm_svm_infer4.py \
  --config-file <PATH_TO_CONFIG>/config.yaml \
  --pretrained-weights <PATH_TO_BACKBONE>/teacher_checkpoint.pth \
  --pretrained-linear-list "<LINEAR_HEAD_CKPT_1>,<LINEAR_HEAD_CKPT_2>,...,<LINEAR_HEAD_CKPT_K>" \
  --test-dataset normal-triage:root=<TEST_DATA_PATH> \
  --batch-size 1 \
  --training-num-classes 3 \
  --svm-model-path <PATH_TO_SVM>/svm_weight.pickle
