#python test.py --model SegNet --channels 3 --model_path ./checkpoints/lung/SegNet.pth --output_path
# ./output/test/ --mode test --attacks DAG_C --adv_model SegNet --data_path lung
#  python test.py --model UNet --channels 3 --model_path ./checkpoints/brain/UNet.pth --output_path \
#  ./output/test/ --mode test --attacks DAG_A --adv_model UNet --data_path brain
#python test_ag.py --best_model ./checkpoints/octafull/AgNet.pth --mode test --data_path octafull --output_path \
#./output/test/ --model AgNet --adv_model UNet --attacks dag --target 1

#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 --channels 1 \
#--data_type org
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 --channels 1 \
#--adv_model SegNet --attacks dag --data_type DAG_A --target 0 --mask_type 1
#
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 --channels 1 \
#--adv_model SegNet --attacks dag --data_type DAG_B  --target 0  --mask_type 2
#
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 --channels 1 \
#--adv_model SegNet --attacks dag --data_type DAG_C --target 1  --mask_type 3

#---------------------
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 --channels 1 \
#--data_type org
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode train --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 --channels 1 \
#--adv_model UNet --attacks dag --data_type DAG_A --target 0 --mask_type 1
#
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 --channels 1 \
#--adv_model UNet --attacks dag --data_type DAG_B --target 0 --mask_type 2
#
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 --channels 1 \
#--adv_model UNet --attacks dag --data_type DAG_C --target 1 --mask_type 3

#---------------------
#python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 --channels 1 \
#--data_type org
python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 --channels 1 \
--adv_model DenseNet --attacks dag --data_type DAG_A  --target 0 --mask_type 1

python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 --channels 1 \
--adv_model DenseNet --attacks dag --data_type DAG_B --target 0  --mask_type 2

python tools/defense_hgd.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 --channels 1 \
--adv_model DenseNet --attacks dag --data_type DAG_C --target 1 --mask_type 3