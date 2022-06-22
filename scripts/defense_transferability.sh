###train on scaling and test on ifgsm and dag###
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 --transfer \
#--attacks dag --data_type DAG_A --device 0 --channels 1 --target 0 --mask_type 1 --suffix pvt_scl_cat_leff -m \
#--denoiser_path ./checkpoints/denoiser/lung/scl_attk/UNet_pvt_scl_cat_leff.pth
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 --transfer \
#--attacks dag --data_type DAG_A --device 0 --channels 1 --target 0 --mask_type 1 --suffix pvt_scl_cat_leff -m \
#--denoiser_path ./checkpoints/denoiser/lung/scl_attk/UNet_pvt_scl_cat_leff.pth
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 --transfer \
#--attacks dag --data_type DAG_A --device 0 --channels 1 --target 0 --mask_type 1 --suffix pvt_scl_cat_leff -m \
#--denoiser_path ./checkpoints/denoiser/lung/scl_attk/UNet_pvt_scl_cat_leff.pth
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --channels 1 --classes 2 --transfer --attacks dag --data_type DAG_A --mask_type 1 --target 0 --suffix pvt_scl_cat_leff -m \
#--denoiser_path ./checkpoints/denoiser/lung/scl_attk/UNet_pvt_scl_cat_leff.pth
#
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 --transfer \
#--attacks dag --data_type DAG_D --device 0 --channels 1 --target 0 --mask_type 4 --suffix pvt_scl_cat_leff -m \
#--denoiser_path ./checkpoints/denoiser/lung/scl_attk/UNet_pvt_scl_cat_leff.pth
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 --transfer \
#--attacks dag --data_type DAG_D --device 0 --channels 1 --target 0 --mask_type 4 --suffix pvt_scl_cat_leff -m \
#--denoiser_path ./checkpoints/denoiser/lung/scl_attk/UNet_pvt_scl_cat_leff.pth
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 --transfer \
#--attacks dag --data_type DAG_D --device 0 --channels 1 --target 0 --mask_type 4 --suffix pvt_scl_cat_leff -m \
#--denoiser_path ./checkpoints/denoiser/lung/scl_attk/UNet_pvt_scl_cat_leff.pth
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --channels 1 --classes 2 --transfer --attacks dag --data_type DAG_D --mask_type 4 --target 0 --suffix pvt_scl_cat_leff -m \
#--denoiser_path ./checkpoints/denoiser/lung/scl_attk/UNet_pvt_scl_cat_leff.pth


#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 --channels 1 \
#--attacks ifgsm --mask_type 1 --target 0 --suffix pvt_scl_cat_leff --transfer -m \
#--denoiser_path ./checkpoints/denoiser/lung/scl_attk/UNet_pvt_scl_cat_leff.pth
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 --channels 1 \
#--attacks ifgsm --device 1 --mask_type 1 --target 0 --suffix pvt_scl_cat_leff --transfer -m \
#--denoiser_path ./checkpoints/denoiser/lung/scl_attk/UNet_pvt_scl_cat_leff.pth
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 --channels 1 \
#--attacks ifgsm --device 1 --mask_type 1 --target 0 --suffix pvt_scl_cat_leff --transfer -m \
#--denoiser_path ./checkpoints/denoiser/lung/scl_attk/UNet_pvt_scl_cat_leff.pth
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --classes 2 --channels 1 --attacks ifgsm  --device 1 --mask_type 1 --target 0 --suffix pvt_scl_cat_leff --transfer -m \
#--denoiser_path ./checkpoints/denoiser/lung/scl_attk/UNet_pvt_scl_cat_leff.pth

###train on dag and test on ifgsm and scaling###
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 --channels 1 \
#--attacks ifgsm --mask_type 1 --target 0 --suffix pvt_dag_cat_leff_3cpn --transfer -m \
#--denoiser_path ./checkpoints/denoiser/lung/dag/UNet_pvt_dag_cat_leff_3cpn.pth
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 --channels 1 \
#--attacks ifgsm --device 1 --mask_type 1 --target 0 --suffix pvt_dag_cat_leff_3cpn --transfer -m \
#--denoiser_path ./checkpoints/denoiser/lung/dag/UNet_pvt_dag_cat_leff_3cpn.pth
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 --channels 1 \
#--attacks ifgsm --device 1 --mask_type 1 --target 0 --suffix pvt_dag_cat_leff_3cpn --transfer -m \
#--denoiser_path ./checkpoints/denoiser/lung/dag/UNet_pvt_dag_cat_leff_3cpn.pth
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --classes 2 --channels 1 --attacks ifgsm  --device 1 --mask_type 1 --target 0 --suffix pvt_dag_cat_leff_3cpn --transfer -m \
#--denoiser_path ./checkpoints/denoiser/lung/dag/UNet_pvt_dag_cat_leff_3cpn.pth

#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 --channels 1 \
#--attacks scl_attk  --suffix pvt_dag_cat_leff_3cpn --transfer -m \
#--denoiser_path ./checkpoints/denoiser/lung/dag/UNet_pvt_dag_cat_leff_3cpn.pth
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 --channels 1 \
#--attacks scl_attk --device 1  --suffix pvt_dag_cat_leff_3cpn --transfer -m \
#--denoiser_path ./checkpoints/denoiser/lung/dag/UNet_pvt_dag_cat_leff_3cpn.pth
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 --channels 1 \
#--attacks scl_attk --device 1  --suffix pvt_dag_cat_leff_3cpn --transfer -m \
#--denoiser_path ./checkpoints/denoiser/lung/dag/UNet_pvt_dag_cat_leff_3cpn.pth
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --classes 2 --channels 1 --attacks scl_attk  --device 1  --suffix pvt_dag_cat_leff_3cpn --transfer -m \
#--denoiser_path ./checkpoints/denoiser/lung/dag/UNet_pvt_dag_cat_leff_3cpn.pth

###train on ifgsm and test on dag and scaling###
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 --channels 1 \
--attacks scl_attk  --suffix pvt_ifgsm_cat_leff_3cpn --transfer -m \
--denoiser_path ./checkpoints/denoiser/lung/ifgsm/UNet_pvt_ifgsm_cat_leff_3cpn.pth
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 --channels 1 \
--attacks scl_attk --device 1  --suffix pvt_ifgsm_cat_leff_3cpn --transfer -m \
--denoiser_path ./checkpoints/denoiser/lung/ifgsm/UNet_pvt_ifgsm_cat_leff_3cpn.pth
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 --channels 1 \
--attacks scl_attk --device 1  --suffix pvt_ifgsm_cat_leff_3cpn --transfer -m \
--denoiser_path ./checkpoints/denoiser/lung/ifgsm/UNet_pvt_ifgsm_cat_leff_3cpn.pth
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
 --classes 2 --channels 1 --attacks scl_attk  --device 1  --suffix pvt_ifgsm_cat_leff_3cpn --transfer -m \
--denoiser_path ./checkpoints/denoiser/lung/ifgsm/UNet_pvt_ifgsm_cat_leff_3cpn.pth

python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 --transfer \
--attacks dag --data_type DAG_A --device 0 --channels 1 --target 0 --mask_type 1 --suffix pvt_scl_cat_leff -m \
--denoiser_path ./checkpoints/denoiser/lung/ifgsm/UNet_pvt_ifgsm_cat_leff_3cpn.pth
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 --transfer \
--attacks dag --data_type DAG_A --device 0 --channels 1 --target 0 --mask_type 1 --suffix pvt_scl_cat_leff -m \
--denoiser_path ./checkpoints/denoiser/lung/ifgsm/UNet_pvt_ifgsm_cat_leff_3cpn.pth
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 --transfer \
--attacks dag --data_type DAG_A --device 0 --channels 1 --target 0 --mask_type 1 --suffix pvt_scl_cat_leff -m \
--denoiser_path ./checkpoints/denoiser/lung/ifgsm/UNet_pvt_ifgsm_cat_leff_3cpn.pth
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
 --channels 1 --classes 2 --transfer --attacks dag --data_type DAG_A --mask_type 1 --target 0 --suffix pvt_scl_cat_leff -m \
--denoiser_path ./checkpoints/denoiser/lung/ifgsm/UNet_pvt_ifgsm_cat_leff_3cpn.pth

python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 --transfer \
--attacks dag --data_type DAG_D --device 0 --channels 1 --target 0 --mask_type 4 --suffix pvt_scl_cat_leff -m \
--denoiser_path ./checkpoints/denoiser/lung/ifgsm/UNet_pvt_ifgsm_cat_leff_3cpn.pth
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 --transfer \
--attacks dag --data_type DAG_D --device 0 --channels 1 --target 0 --mask_type 4 --suffix pvt_scl_cat_leff -m \
--denoiser_path ./checkpoints/denoiser/lung/ifgsm/UNet_pvt_ifgsm_cat_leff_3cpn.pth
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 --transfer \
--attacks dag --data_type DAG_D --device 0 --channels 1 --target 0 --mask_type 4 --suffix pvt_scl_cat_leff -m \
--denoiser_path ./checkpoints/denoiser/lung/ifgsm/UNet_pvt_ifgsm_cat_leff_3cpn.pth
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
 --channels 1 --classes 2 --transfer --attacks dag --data_type DAG_D --mask_type 4 --target 0 --suffix pvt_scl_cat_leff -m \
--denoiser_path ./checkpoints/denoiser/lung/ifgsm/UNet_pvt_ifgsm_cat_leff_3cpn.pth
