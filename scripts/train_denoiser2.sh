#!/bin/bash

#python tools/train_denoiser_semantic.py \
#        --data_path octafull --model UNet --classes 3 --channels 1  --batch_size 2 --suffix pvt_semantic_plus_leff  --device 0

#python tools/train_denoiser_semantic.py \
#        --data_path octafull --model UNet --classes 3 --channels 1  --batch_size 2 --suffix pvt_semantic_plus_leff_sub  --device 0

#python tools/train_denoiser_semantic.py --data_type semantic \
#        --data_path brain --model UNet --classes 2 --channels 3  --batch_size 4 --suffix pvt_semantic_plus_leff  --device 1

python tools/train_denoiser_semantic.py --data_type semantic\
        --data_path fundus --model UNet --classes 3 --channels 3  --batch_size 4 --suffix pvt_semantic_plus_leff  --device 0

#python tools/train_denoiser.py \
#        --data_path octafull --model UNet --classes 3 --channels 1  --batch_size 4 --suffix rd  --device 0


