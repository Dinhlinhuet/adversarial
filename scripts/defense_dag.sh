#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
#--attacks semantic --data_type org --device 0 --channels 1
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
#--attacks semantic --data_type org --device 0 --channels 1
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
#--attacks semantic --data_type org --device 0 --channels 1


#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
#--attacks semantic --data_type DAG_A --device 0 --channels 1 --target 0 --mask_type 1
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
#--attacks semantic --data_type DAG_A --device 0 --channels 1 --target 0 --mask_type 1
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
#--attacks semantic --data_type DAG_A --device 0 --channels 1 --target 0 --mask_type 1

#python tools/defense.py --model_path ./checkpoints/ --mode train --data_path octafull --output_path \
#./output/train/denoise/ --denoise_output ./output/train/denoised_imgs/ --model SegNet --classes 3 \
#--attacks semantic --data_type DAG_A --device 0 --channels 1 --target 0 --mask_type 1
python tools/defense.py --model_path ./checkpoints/ --mode train --data_path octafull --output_path \
./output/train/denoise/ --denoise_output ./output/train/denoised_imgs/ --model UNet --classes 3 \
--attacks dag --data_type DAG_A --device 0 --channels 1 --target 0 --mask_type 1
#python tools/defense.py --model_path ./checkpoints/ --mode train --data_path octafull --output_path \
#./output/train/denoise/ --denoise_output ./output/train/denoised_imgs/ --model DenseNet --classes 3 \
#--attacks semantic --data_type DAG_A --device 0 --channels 1 --target 0 --mask_type 1

#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
#--attacks semantic --data_type DAG_B --device 0 --channels 1 --target 0 --mask_type 2
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
#--attacks semantic --data_type DAG_B --device 0 --channels 1 --target 0 --mask_type 2
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
#--attacks semantic --data_type DAG_B --device 0 --channels 1 --target 0 --mask_type 2
#
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
#--attacks semantic --data_type DAG_C --device 0 --channels 1 --target 1 --mask_type 3
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
#--attacks semantic --data_type DAG_C --device 0 --channels 1 --target 1 --mask_type 3
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
#--attacks semantic --data_type DAG_C --device 0 --channels 1 --target 1 --mask_type 3