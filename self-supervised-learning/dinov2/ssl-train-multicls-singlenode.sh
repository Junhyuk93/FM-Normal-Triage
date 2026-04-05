torchrun --nproc_per_node=8 dinov2/train/train.py \
    --config-file /home/srpark/self-supervised-learning/dinov2/dinov2/configs/ssl_our_config.yaml \
    --output-dir ./results \
    train.dataset_path=PD:root=/home/dataset/FM_dataset

