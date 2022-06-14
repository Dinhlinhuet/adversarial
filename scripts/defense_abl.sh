#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
#--attacks scl_attk --device 1 --suffix pvt_scl_naive --source_version 1
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
#--attacks scl_attk --suffix pvt_scl_naive --source_version 1
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
#--attacks scl_attk --suffix pvt_scl_naive --source_version 1
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 3 \
#--attacks scl_attk --suffix pvt_scl_naive --source_version 1
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --classes 3 --attacks scl_attk --suffix pvt_scl_naive --source_version 1

#CUDA_LAUNCH_BLOCKING=1 python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
#--attacks scl_attk --device 1 --suffix pvt_scl_cat_leff --source_version 1 -m
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
#--attacks scl_attk --suffix pvt_scl_cat_leff --source_version 1 -m
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
#--attacks scl_attk --suffix pvt_scl_cat_leff --source_version 1 -m
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 3 \
#--attacks scl_attk --suffix pvt_scl_cat_leff --source_version 1 -m
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --classes 3 --attacks scl_attk --suffix pvt_scl_cat_leff --source_version 1 -m

#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
#--attacks scl_attk --data_type org --suffix pvt_scl_naive
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
#--attacks scl_attk --data_type org --suffix pvt_scl_naive
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
#--attacks scl_attk --data_type org --suffix pvt_scl_naive
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 3 \
#--attacks scl_attk --data_type org --suffix pvt_scl_naive
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --classes 3 --attacks scl_attk --data_type org --suffix pvt_scl_naive

#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 \
#--attacks dag --data_type DAG_A --device 0 --channels 1 --target 0 --mask_type 1 --suffix pvt_dag_naive
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 \
#--attacks dag --data_type DAG_A --device 0 --channels 1 --target 0 --mask_type 1 --suffix pvt_dag_naive
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 \
#--attacks dag --data_type DAG_A --device 0 --channels 1 --target 0 --mask_type 1 --suffix pvt_dag_naive
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --channels 1 --classes 2 --attacks dag --data_type DAG_A --mask_type 1 --target 0 --suffix pvt_dag_naive
#
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 \
#--attacks dag --data_type DAG_D --device 0 --channels 1 --target 0 --mask_type 4 --suffix pvt_dag_naive
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 \
#--attacks dag --data_type DAG_D --device 0 --channels 1 --target 0 --mask_type 4 --suffix pvt_dag_naive
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 \
#--attacks dag --data_type DAG_D --device 0 --channels 1 --target 0 --mask_type 4 --suffix pvt_dag_naive
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --channels 1 --classes 2 --attacks dag --data_type DAG_D --mask_type 4 --target 0 --suffix pvt_dag_naive

#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 \
#--attacks dag --data_type org --device 0 --channels 1 --target 0 --suffix pvt_dag_naive
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 \
#--attacks dag --data_type org --device 0 --channels 1 --target 0 --suffix pvt_dag_naive
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 \
#--attacks dag --data_type org --device 0 --channels 1 --target 0 --suffix pvt_dag_naive
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --channels 1 --classes 2 --attacks dag --data_type org --target 0 --suffix pvt_dag_naive

#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 2 \
#--attacks scl_attk --device 1 --suffix pvt_scl_plus_leff --source_version 0 -m
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 \
#--attacks scl_attk --device 1 --suffix pvt_scl_plus_leff --source_version 0 -m
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 \
#--attacks scl_attk --device 1 --suffix pvt_scl_plus_leff --source_version 0 -m
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_mobilenet --output_stride 8\
# --classes 2 --attacks scl_attk  --device 1 --suffix pvt_scl_plus_leff --source_version 0 -m

#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 2 \
#--attacks scl_attk --device 1 --suffix pvt_scl_plus_leff --source_version 2 -m
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 \
#--attacks scl_attk --device 1 --suffix pvt_scl_plus_leff --source_version 2 -m
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 \
#--attacks scl_attk --device 1 --suffix pvt_scl_plus_leff --source_version 2 -m
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_mobilenet --output_stride 8\
# --classes 2 --attacks scl_attk  --device 1 --suffix pvt_scl_plus_leff --source_version 2 -m

#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 2 \
#--attacks scl_attk --device 1 --suffix pvt_scl_naive --source_version 2
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 \
#--attacks scl_attk --device 1 --suffix pvt_scl_naive --source_version 2
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 \
#--attacks scl_attk --device 1 --suffix pvt_scl_naive --source_version 2
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_mobilenet --output_stride 8\
# --classes 2 --attacks scl_attk  --device 1 --suffix pvt_scl_naive --source_version 2
 
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 2 \
--attacks scl_attk --device 1 --suffix pvt_scl_cat_foursome_loss --source_version 0 -m
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 \
--attacks scl_attk --device 1 --suffix pvt_scl_cat_foursome_loss --source_version 0 -m
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 \
--attacks scl_attk --device 1 --suffix pvt_scl_cat_foursome_loss --source_version 0 -m
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_mobilenet --output_stride 8\
 --classes 2 --attacks scl_attk  --device 1 --suffix pvt_scl_cat_foursome_loss --source_version 0 -m