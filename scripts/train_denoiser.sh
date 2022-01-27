#!/bin/bash
#python train_denoiser.py \
#        --data_path fundus --model UNet --classes 3 --channels 3  --batch_size 1 --suffix rd --nodes 1 --gpus 1\
#        --device2 1
#python train_denoiser.py \
#        --data_path fundus --model DenseNet --classes 3 --channels 3  --batch_size 2 --suffix trf_rd  --gpus 1
#python train_denoiser_new.py \
#        --data_path fundus --model DenseNet --classes 3 --channels 3  --batch_size 2 --suffix trf_rd  --gpus 2 \
#        --device2 1
#python train_denoiser_new.py \
#        --data_path fundus --model UNet --classes 3 --channels 3  --batch_size 2 --suffix trf_rd  --gpus 1 \

#python train_ag_denoiser_new.py \
#        --data_path fundus --model AgNet --classes 3 --channels 3  --batch_size 4 --suffix trf_rd  --gpus 1 \

#python train_agdenoiser.py \
#        --data_path fundus --model AgNet --classes 3 --channels 3  --batch_size 2 --suffix rd  --gpus 1
#rd means random
#python train_denoiser.py \
#        --data_path octa3mfull --model DenseNet --classes 3 --channels 1  --batch_size 1 --suffix rd  --gpus 1

python tools/train_denoiser_scl.py \
        --data_path fundus --model UNet --classes 3 --channels 3  --batch_size 4 --suffix pvt_scl  --device 0

#python tools/train_denoiser_scl.py \
#        --data_path fundus --model UNet --classes 3 --channels 3  --batch_size 4 --suffix pvt  --device 1