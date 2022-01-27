#!/bin/bash
#python train_denoiser.py \
#        --data_path fundus --model UNet --classes 3 --channels 3  --batch_size 1 --suffix attent --nodes 1 --gpus 2\
#        --device2 1

#python train.py --model SegNet --data_path octafull --classes 3 --channels 1 --device 0
#python train.py --model UNet --data_path octafull --classes 3 --channels 1 --device 0
#python train.py --model DenseNet --data_path octafull --classes 3 --channels 1 --device 0 --batch_size 4


#python tools/train_deeplab.py --model deeplabv3plus_mobilenet  --data_path fundus --device 0 --lr 0.01 --batch_size 16\
# --output_stride 16 --classes 3 --channels 3

#python tools/train_deeplab.py --model deeplabv3plus_hrnetv2_48  --data_path fundus --device 0 --lr 0.01 --batch_size 16\
# --output_stride 8 --classes 3 --channels 3

#python tools/train_deeplab.py --model deeplabv3plus_resnet101  --data_path fundus --device 0 --lr 0.01 --batch_size 16\
# --output_stride 8 --classes 3 --channels 3

#python tools/train_deeplab.py --model deeplabv3_mobilenet_v3_large  --data_path fundus --device 0 --lr 0.01 --batch_size 16\
# --output_stride 8 --classes 3 --channels 3

#python tools/train.py --model CENet --data_path fundus --classes 3 --channels 3 --device 0 --batch_size 4

#python tools/train_deeplab.py --model deeplabv3plus_mobilenet  --data_path brain --device 0 --lr 0.01 --batch_size 8\
# --output_stride 8 --classes 2 --channels 3

python tools/train_deeplab.py --model deeplabv3plus_resnet101  --data_path octafull --device 0 --lr 0.01 --batch_size 16\
 --output_stride 8 --classes 3 --channels 1