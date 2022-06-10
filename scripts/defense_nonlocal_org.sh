##scl attack##
##fundus##
#python tools/defense_nonlocal.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
#--attacks scl_attk --data_type org
#python tools/defense_nonlocal.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
#--attacks scl_attk --data_type org
#python tools/defense_nonlocal.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
#--attacks scl_attk --data_type org
#python tools/defense_nonlocal.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 3 \
#--attacks scl_attk --data_type org --batch_size 1
#python tools/defense_nonlocal.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101\
# --classes 3 --attacks scl_attk --data_type org
##brain##
#python tools/defense_nonlocal.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 2 \
#--attacks scl_attk --data_type org --device 0
#python tools/defense_nonlocal.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 \
#--attacks scl_attk --data_type org --device 0
#python tools/defense_nonlocal.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 \
#--attacks scl_attk --data_type org --device 0
#python tools/defense_nonlocal.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_mobilenet\
# --classes 2 --attacks scl_attk --data_type org  --device 0
##lung##
#python tools/defense_nonlocal.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 --channels 1 \
#--attacks scl_attk --data_type org --batch_size 1
#python tools/defense_nonlocal.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 --channels 1 \
#--attacks scl_attk --data_type org
#python tools/defense_nonlocal.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 --channels 1 \
#--attacks scl_attk --data_type org
#python tools/defense_nonlocal.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101\
# --classes 2 --channels 1 --attacks scl_attk --data_type org

##ifgsm##
python tools/defense_nonlocal.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 \
--attacks ifgsm --channels 1 --data_type org --device 1
#python tools/defense_nonlocal.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 \
#--attacks ifgsm --channels 1 --data_type org
#python tools/defense_nonlocal.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 \
#--attacks ifgsm --channels 1 --data_type org
#python tools/defense_nonlocal.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101\
# --classes 2 --attacks ifgsm --channels 1 --data_type org

##dag##
python tools/defense_nonlocal.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 \
--attacks dag --channels 1 --data_type org --device 1
#python tools/defense_nonlocal.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 \
#--attacks dag --channels 1 --data_type org --device 0 --channels 1
#python tools/defense_nonlocal.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 \
#--attacks dag --channels 1 --data_type org --device 0 --channels 1
#python tools/defense_nonlocal.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101\
# --channels 1 --classes 2 --attacks dag --data_type org

