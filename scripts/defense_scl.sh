python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
--attacks scl_attk
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
--attacks scl_attk
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
--attacks scl_attk
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 3 \
--attacks scl_attk
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
 --classes 3 --attacks scl_attk