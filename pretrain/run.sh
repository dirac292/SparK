
#!/bin/bash

base_dir_0='/mnt/pub0'
base_dir_1='/mnt/pub1'
log_dir="${base_dir_0}/ssl-pretraining/logs"
data_dir="${base_dir_1}/ssl-pretraining/data"
poly_pretrain_dir='/mnt/pub1/ssl-pretraining/data/hyper-kvasir/unlabeled-images/train/images'
hist_dir='/mnt/pub1/ssl-pretraining/data/histopathology'
hist_pretrain_dir="${hist_dir}/aSMA_SmoothMuscle/Images"

torchrun --nproc_per_node=2 --master_port=29500 main.py \
  --exp_name spark_dicoh_match \
  --exp_dir $log_dir/logs_spark \
  --data_path /mnt/pub1/ssl-pretraining/data/hyper-kvasir/unlabeled-images/images/train \
  --model resnet50 \
  --input_size 224 \
  --bs 128 \
  --ep 10 \
  --opt sgd \
  --base_lr 0.002 \
  --wd 0.0001 \
  --wde 0.0001 \
  --dataloader_workers 32 \
  --mask 0.6
