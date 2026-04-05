python fm_svm_train.py \
  --config-file <PATH_TO_CONFIG>/config.yaml \
  --pretrained-weights <PATH_TO_BACKBONE>/teacher_checkpoint.pth \
  --pretrained-linear-list "<LINEAR_HEAD_CKPT_1>,<LINEAR_HEAD_CKPT_2>,...,<LINEAR_HEAD_CKPT_K>" \
  --train-dataset normal-triage:root=<TRAIN_DATA_PATH> \
  --batch-size 1 \
  --training-num-classes 3 \
  --svm-type rbf \
  --C 1.0 \
  --svm-out <OUTPUT_DIR>/svm_weight.pickle
