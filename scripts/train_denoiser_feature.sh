##fundus##
#python tools/train_denoiser_feature.py \
#        --data_path fundus --model UNet --classes 3 --channels 3  --batch_size 1 --suffix feature \
#        --device 1 --data_type scl_attk

#python tools/train_denoiser_feature.py \
#        --data_path fundus --model SegNet --classes 3 --channels 3  --batch_size 1 --suffix feature \
#        --device 0 --data_type scl_attk

#python tools/train_denoiser_feature.py \
#        --data_path fundus --model DenseNet --classes 3 --channels 3  --batch_size 1 --suffix feature \
#        --device 1 --data_type scl_attk

#python tools/train_denoiser_feature.py \
#        --data_path fundus --model AgNet --classes 3 --channels 3  --batch_size 1 --suffix feature \
#        --device 0 --data_type scl_attk

#python tools/train_denoiser_feature.py \
#        --data_path fundus --model deeplabv3plus_resnet101 --classes 3 --channels 3  --batch_size 2 --suffix feature \
#        --device 1 --data_type scl_attk

##brain##
#python tools/train_denoiser_feature.py \
#        --data_path brain --model UNet --classes 2 --channels 3  --batch_size 1 --suffix feature \
#        --device 0 --data_type scl_attk
#
#python tools/train_denoiser_feature.py \
#        --data_path brain --model SegNet --classes 2 --channels 3  --batch_size 1 --suffix feature \
#        --device 0 --data_type scl_attk
#
#python tools/train_denoiser_feature.py \
#        --data_path brain --model DenseNet --classes 2 --channels 3  --batch_size 1 --suffix feature \
#        --device 0 --data_type scl_attk
#
#python tools/train_denoiser_feature.py \
#        --data_path brain --model deeplabv3plus_mobilenet --classes 2 --channels 3  --batch_size 2 --suffix feature \
#        --device 0 --data_type scl_attk

##lung##
#python tools/train_denoiser_feature.py \
#        --data_path lung --model UNet --classes 2 --channels 1 --batch_size 1 --suffix feature \
#        --device 0 --data_type scl_attk

#python tools/train_denoiser_feature.py \
#        --data_path lung --model AgNet --classes 2 --channels 1 --batch_size 1 --suffix feature \
#        --device 0 --data_type scl_attk

#python tools/train_denoiser_feature.py \
#        --data_path lung --model DenseNet --classes 2 --channels 1 --batch_size 1 --suffix feature \
#        --device 0 --data_type scl_attk
#
#python tools/train_denoiser_feature.py \
#        --data_path lung --model deeplabv3plus_resnet101 --classes 2 --channels 1 --batch_size 2 --suffix feature \
#        --device 0 --data_type scl_attk

#dag#

python tools/train_denoiser_feature.py \
        --data_path lung --model AgNet --classes 2 --channels 1 --batch_size 1 --suffix feature \
        --device 1 --data_type dag

#python tools/train_denoiser_feature.py \
#        --data_path lung --model UNet --classes 2 --channels 1 --batch_size 1 --suffix feature \
#        --device 0 --data_type dag
#
#python tools/train_denoiser_feature.py \
#        --data_path lung --model DenseNet --classes 2 --channels 1 --batch_size 1 --suffix feature \
#        --device 0 --data_type dag
#
#python tools/train_denoiser_feature.py \
#        --data_path lung --model deeplabv3plus_resnet101 --classes 2 --channels 1 --batch_size 2 --suffix feature \
#        --device 0 --data_type dag

#ifgsm#
#python tools/train_denoiser_feature.py \
#        --data_path lung --model AgNet --classes 2 --channels 1 --batch_size 1 --suffix feature \
#        --device 0 --data_type ifgsm
#
#python tools/train_denoiser_feature.py \
#        --data_path lung --model UNet --classes 2 --channels 1 --batch_size 1 --suffix feature \
#        --device 0 --data_type ifgsm

#python tools/train_denoiser_feature.py \
#        --data_path lung --model DenseNet --classes 2 --channels 1 --batch_size 1 --suffix feature \
#        --device 1 --data_type ifgsm
#
#python tools/train_denoiser_feature.py \
#        --data_path lung --model deeplabv3plus_resnet101 --classes 2 --channels 1 --batch_size 2 --suffix feature \
#        --device 1 --data_type ifgsm