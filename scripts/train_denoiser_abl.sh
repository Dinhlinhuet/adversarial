#python tools/train_denoiser_scl.py \
#        --data_path fundus --model UNet --classes 3 --channels 3  --batch_size 4 --suffix pvt_scl_naive  --device 0

#python tools/train_denoiser_semantic.py --data_type dag\
#        --data_path lung --model UNet --classes 2 --channels 1  --batch_size 4 --suffix pvt_dag_naive  --device 0

#python tools/train_denoiser_scl.py \
#        --data_path brain --model UNet --classes 2 --channels 3  --batch_size 4 --suffix pvt_scl_naive  --device 1

python tools/train_denoiser_scl.py \
        --data_path brain --model UNet --classes 2 --channels 3  --batch_size 4 --suffix pvt_scl_cat_foursome_loss  \
        -m --device 1
