#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 \
#--attacks ifgsm --mask_type 1 --target 0
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 \
#--attacks ifgsm --mask_type 1 --target 0
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 \
#--attacks ifgsm --mask_type 1 --target 0
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --classes 2 --attacks ifgsm --mask_type 1 --target 0

#### for transformer model###
###
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 --channels 1 \
--attacks ifgsm --device 1 --mask_type 1 --target 0 --suffix pvt_ifgsm_cat_leff_3cpn
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 --channels 1 \
--attacks ifgsm --device 1 --mask_type 1 --target 0 --suffix pvt_ifgsm_cat_leff_3cpn
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 --channels 1 \
#--attacks ifgsm --mask_type 1 --target 0 --suffix pvt_ifgsm_cat_leff_3cpn
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --classes 2 --channels 1 --attacks ifgsm  --device 1 --mask_type 1 --target 0 --suffix pvt_ifgsm_cat_leff_3cpn

#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 --channels 1 \
#--attacks ifgsm --device 1 --data_type org --suffix pvt_ifgsm_cat_leff_3cpn
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 --channels 1 \
#--attacks ifgsm --device 1 --data_type org --suffix pvt_ifgsm_cat_leff_3cpn
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 --channels 1 \
#--attacks ifgsm --data_type org --suffix pvt_ifgsm_cat_leff_3cpn
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --classes 2 --channels 1 --attacks ifgsm  --device 1 --data_type org --suffix pvt_ifgsm_cat_leff_3cpn

###for HGD model###
###attack fgsm##
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 \
#--attacks ifgsm --mask_type 1 --target 0 --suffix hgd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 \
#--attacks ifgsm --mask_type 1 --target 0 --suffix hgd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 \
#--attacks ifgsm --mask_type 1 --target 0 --suffix hgd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --classes 2 --attacks ifgsm --mask_type 1 --target 0 --suffix hgd

## for org data##
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 --channels 1 \
#--attacks ifgsm --data_type org --suffix hgd_rd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 --channels 1 \
#--attacks ifgsm --data_type org --suffix hgd_rd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 --channels 1 \
#--attacks ifgsm --data_type org --suffix hgd_rd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --classes 2 --channels 1 --attacks ifgsm --data_type org --suffix hgd_rd


##attack dag###
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 --channels 1 \
#--attacks dag --device 1 --data_type DAG_A --mask_type 1 --target 0 --suffix rd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 --channels 1 \
#--attacks dag --device 1 --data_type DAG_A --mask_type 1 --target 0 --suffix rd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 --channels 1 \
#--attacks dag --device 1 --data_type DAG_A --mask_type 1 --target 0 --suffix rd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --classes 2 --channels 1 --attacks dag --data_type DAG_A  --device 1 --mask_type 1 --target 0 --suffix rd
#
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 --channels 1 \
#--attacks dag --device 1 --data_type DAG_D --mask_type 4 --target 0 --suffix rd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 --channels 1 \
#--attacks dag --device 1 --data_type DAG_D --mask_type 4 --target 0 --suffix rd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 --channels 1 \
#--attacks dag --device 1 --data_type DAG_D --mask_type 4 --target 0 --suffix rd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --channels 1 --classes 2 --attacks dag --data_type DAG_D  --device 1 --mask_type 4 --target 0 --suffix rd

#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 --channels 1 \
#--attacks dag --device 1 --data_type org  --suffix rd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 --channels 1 \
#--attacks dag --device 1 --data_type org  --suffix rd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 --channels 1 \
#--attacks dag --device 1 --data_type org  --suffix rd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --classes 2 --channels 1 --attacks dag --data_type org  --device 1  --suffix rd

##attack ifgsm###
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 --channels 1 \
#--attacks ifgsm --device 1 --mask_type 1 --target 0 --suffix rd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 --channels 1 \
#--attacks ifgsm --device 1 --mask_type 1 --target 0 --suffix rd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 --channels 1 \
#--attacks ifgsm --device 1 --mask_type 1 --target 0 --suffix rd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --classes 2 --channels 1 --attacks ifgsm  --device 1 --mask_type 1 --target 0 --suffix rd