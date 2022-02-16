#!/bin/bash
#python train_denoiser.py \
#        --data_path fundus --model UNet --classes 3 --channels 3  --batch_size 1 --suffix rd --nodes 1 --gpus 1\
#        --device2 1
#python train_denoiser.py \
#        --data_path fundus --model DenseNet --classes 3 --channels 3  --batch_size 2 --suffix trf_rd  --gpus 1

#python train_denoiser.py \
#        --data_path octa3mfull --model DenseNet --classes 3 --channels 1  --batch_size 1 --suffix rd  --gpus 1

#python tools/train_denoiser_scl.py \
#        --data_path fundus --model UNet --classes 3 --channels 3  --batch_size 4 --suffix pvt_scl  --device 0

#python tools/train_denoiser_scl.py \
#        --data_path fundus --model UNet --classes 3 --channels 3  --batch_size 4 --suffix pvt  --device 1

python tools/train_denoiser_scl.py \
        --data_path fundus --model UNet --classes 3 --channels 3  --batch_size 4 --suffix pvt_scl_plus_leff  --device 0


#python tools/train_denoiser_scl.py \
#        --data_path brain --model UNet --classes 2 --channels 3  --batch_size 4 --suffix pvt_scl  --device 0

#python tools/train_denoiser_scl.py \
#        --data_path brain --model UNet --classes 2 --channels 3  --batch_size 4 --suffix pvt_scl_plus  --device 0

#python tools/train_denoiser_scl.py \
#        --data_path brain --model UNet --classes 2 --channels 3  --batch_size 4 --suffix pvt_scl_plus_leff  --device 0

#python tools/train_denoiser_scl.py --data_path octafull --model UNet --classes 3 --channels 1  --batch_size 2 \
#--suffix pvt_scl_triplet  --device 0 --lr 1e-3

#python tools/train_denoiser_scl.py \
#        --data_path octafull --model UNet --classes 3 --channels 1  --batch_size 4 --suffix pvt_scl_loss_2_sub  --device 0

#python tools/train_denoiser_scl.py \
#        --data_path octafull --model UNet --classes 3 --channels 1  --batch_size 4 --suffix pvt_scl_plus_leff  --device 1

