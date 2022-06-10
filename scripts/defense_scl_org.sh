#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
#--attacks scl_attk --data_type org --suffix pvt_scl_cat_leff
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
#--attacks scl_attk --data_type org --suffix pvt_scl_cat_leff
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
#--attacks scl_attk --data_type org --suffix pvt_scl_cat_leff
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 3 \
#--attacks scl_attk --data_type org --suffix pvt_scl_cat_leff
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --classes 3 --attacks scl_attk --data_type org --suffix pvt_scl_cat_leff
 
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 2 \
#--attacks scl_attk --data_type org --device 1 --suffix pvt_scl_plus_leff
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 \
#--attacks scl_attk --data_type org --device 1 --suffix pvt_scl_plus_leff
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 \
#--attacks scl_attk --data_type org --device 1 --suffix pvt_scl_plus_leff
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_mobilenet --output_stride 8\
# --classes 2 --attacks scl_attk --data_type org  --device 1 --suffix pvt_scl_plus_leff
 
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
#--attacks scl_attk --data_type org --device 1 --channels 1 --suffix pvt_scl_plus_leff
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
#--attacks scl_attk --data_type org --device 1 --channels 1 --suffix pvt_scl_plus_leff
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
#--attacks scl_attk --data_type org --device 1 --channels 1 --suffix pvt_scl_plus_leff


#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 --channels 1 \
#--attacks scl_attk --data_type org --suffix pvt_scl_cat_leff
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 --channels 1 \
#--attacks scl_attk --data_type org --suffix pvt_scl_cat_leff
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 --channels 1 \
#--attacks scl_attk --data_type org --suffix pvt_scl_cat_leff
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 16\
# --classes 2 --channels 1 --attacks scl_attk --data_type org --suffix pvt_scl_cat_leff


## for hgd model###
python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
--attacks scl_attk --data_type org --device 1 --channels 1 --suffix hgd
python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
--attacks scl_attk --data_type org --device 1 --channels 1 --suffix hgd
python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
--attacks scl_attk --data_type org --device 1 --channels 1 --suffix hgd

#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
#--attacks scl_attk --data_type org --device 0 --channels 3 --suffix hgd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
#--attacks scl_attk --data_type org --device 0 --channels 3 --suffix hgd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
#--attacks scl_attk --data_type org --device 0 --channels 3 --suffix hgd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 3 \
#--attacks scl_attk --data_type org --device 0 --channels 3 --suffix hgd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model deeplabv3plus_resnet101 --classes 3 \
# --output_stride 8 --attacks scl_attk --data_type org --device 0 --channels 3 --suffix hgd

#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 2 \
#--attacks scl_attk --data_type org --device 1 --channels 3 --suffix hgd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 \
#--attacks scl_attk --data_type org --device 1 --channels 3 --suffix hgd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 \
#--attacks scl_attk --data_type org --device 1 --channels 3 --suffix hgd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model deeplabv3plus_mobilenet --classes 2 \
# --output_stride 8 --attacks scl_attk --data_type org --device 1 --channels 3 --suffix hgd

#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 \
#--attacks scl_attk --data_type org --device 1 --channels 1 --suffix hgd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 \
#--attacks scl_attk --data_type org --device 1 --channels 1 --suffix hgd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 \
#--attacks scl_attk --data_type org --device 1 --channels 1 --suffix hgd
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model deeplabv3plus_resnet101 --classes 2 \
# --output_stride 8 --attacks scl_attk --data_type org --device 1 --channels 1 --suffix hgd