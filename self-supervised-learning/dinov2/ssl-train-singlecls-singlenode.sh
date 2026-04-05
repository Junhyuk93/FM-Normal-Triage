torchrun --nproc_per_node=2 dinov2/train/train.py \
    --config-file /home/work/code/self-supervised-learning/dinov2/dinov2/configs/ssl_our_config.yaml \
    --output-dir ./output_test\
    train.dataset_path=PD:root=/home/work/dataset/test-cxr

