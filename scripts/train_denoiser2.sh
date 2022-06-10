#!/bin/bash

####model pvt###
#python tools/train_denoiser_semantic.py --data_type dag\
#        --data_path octafull --model UNet --classes 3 --channels 1  --batch_size 2 --suffix pvt_dag_cat_leff  --device 1

#python tools/train_denoiser_semantic.py --data_type dag\
#        --data_path octafull --model UNet --classes 3 --channels 1  --batch_size 2 --suffix pvt_dag_add_leff_3cpn  --device 1

#python tools/train_denoiser_semantic.py --data_type ifgsm \
#        --data_path octafull --model UNet --classes 3 --channels 1  --batch_size 4 --suffix pvt_ifgsm_cat_plus_leff  --device 1

#python tools/train_denoiser_semantic.py --data_type dag\
#        --data_path lung --model UNet --classes 2 --channels 1  --batch_size 2 --suffix pvt_dag_cat_leff_3cpn  --device 0

#python tools/train_denoiser_semantic.py --data_type ifgsm\
#        --data_path lung --model UNet --classes 2 --channels 1  --batch_size 2 --suffix pvt_dag_cat_leff_3cpn  --device 0

#python tools/train_denoiser_semantic.py --data_type dag\
#        --data_path lung --model UNet --classes 2 --channels 1  --batch_size 2 --suffix pvt_dag_cat_leff_conv_3cpn  --device 1

#python tools/train_denoiser_semantic.py --data_type cw\
#        --data_path lung --model UNet --classes 2 --channels 1  --batch_size 2 --suffix pvt_cw_cat_leff_3cpn  --device 1

####model hgd###
#data type = dag
#python tools/train_denoiser.py \
#        --data_path octafull --model UNet --classes 3 --channels 1  --batch_size 4 --suffix rd  --device 0
#python tools/train_denoiser.py --data_type cw\
#        --data_path octafull --model UNet --classes 3 --channels 1  --batch_size 4 --suffix rd  --device 1

#python tools/train_denoiser.py --data_type dag\
#        --data_path lung --model UNet --classes 2 --channels 1  --batch_size 4 --suffix rd  --device 1

#python tools/train_denoiser.py --data_type ifgsm\
#        --data_path lung --model UNet --classes 2 --channels 1  --batch_size 4 --suffix rd  --device 0

### scaling attack##
#python tools/train_denoiser.py --data_type scl_attk\
#        --data_path octafull --model UNet --classes 3 --channels 1  --batch_size 4 --suffix hgd  --device 1

#python tools/train_denoiser.py --data_type scl_attk\
#        --data_path brain --model UNet --classes 2 --channels 3  --batch_size 4 --suffix hgd  --device 1

#python tools/train_denoiser.py --data_type scl_attk\
#        --data_path fundus --model UNet --classes 3 --channels 3  --batch_size 4 --suffix hgd  --device 0

python tools/train_denoiser.py --data_type scl_attk\
        --data_path lung --model UNet --classes 2 --channels 1  --batch_size 4 --suffix hgd  --device 1



