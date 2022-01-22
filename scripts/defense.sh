#python test.py --model SegNet --channels 3 --model_path ./checkpoints/lung/SegNet.pth --output_path
# ./output/test/ --mode test --attacks DAG_C --adv_model SegNet --data_path lung
#  python test.py --model UNet --channels 3 --model_path ./checkpoints/brain/UNet.pth --output_path \
#  ./output/test/ --mode test --attacks DAG_A --adv_model UNet --data_path brain
#python test_ag.py --best_model ./checkpoints/fundus/AgNet.pth --mode test --data_path fundus --output_path \
#./output/test/ --model AgNet --adv_model UNet --attacks ifgsm --target 1

#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
#--data_type org
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
#--adv_model SegNet --attacks ifgsm --target 1 --mask_type 1
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
#--adv_model UNet --attacks ifgsm --target 1 --mask_type 1
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
#--adv_model DenseNet --attacks ifgsm --target 1 --mask_type 1
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
#--adv_model AgNet --attacks ifgsm --target 1 --mask_type 1
#
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
#--adv_model SegNet --attacks ifgsm --target 2
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
#--adv_model UNet --attacks ifgsm --target 2
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
#--adv_model DenseNet --attacks ifgsm --target 2
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
#--adv_model AgNet --attacks ifgsm --target 2
#---------------------
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
#--data_type org
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
#--adv_model SegNet --attacks ifgsm --target 1 --mask_type 1
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
#--adv_model UNet --attacks ifgsm --target 1 --mask_type 1
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
#--adv_model DenseNet --attacks ifgsm --target 1 --mask_type 1
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
#--adv_model AgNet --attacks ifgsm --target 1 --mask_type 1

#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
#--adv_model SegNet --attacks ifgsm --target 2
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
#--adv_model UNet --attacks ifgsm --target 2
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
#--adv_model DenseNet --attacks ifgsm --target 2
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
#--adv_model AgNet --attacks ifgsm --target 2
#---------------------
python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
--data_type org
python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
--adv_model SegNet --attacks ifgsm --target 1 --mask_type 1
python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
--adv_model UNet --attacks ifgsm --target 1 --mask_type 1
python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
--adv_model DenseNet --attacks ifgsm --target 1 --mask_type 1
python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
--adv_model AgNet --attacks ifgsm --target 1 --mask_type 1

python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
--adv_model SegNet --attacks ifgsm --target 2
python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
--adv_model UNet --attacks ifgsm --target 2
python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
--adv_model DenseNet --attacks ifgsm --target 2
python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
--adv_model AgNet --attacks ifgsm --target 2
#---------------------
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 3 \
#--data_type org
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 3 \
#--adv_model SegNet --attacks ifgsm --target 1 --mask_type 1
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 3 \
#--adv_model UNet --attacks ifgsm --target 1 --mask_type 1
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 3 \
#--adv_model DenseNet --attacks ifgsm --target 1 --mask_type 1
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 3 \
#--adv_model AgNet --attacks ifgsm --target 1 --mask_type 1
#
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 3 \
#--adv_model SegNet --attacks ifgsm --target 2
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 3 \
#--adv_model UNet --attacks ifgsm --target 2
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 3 \
#--adv_model DenseNet --attacks ifgsm --target 2
#python defense.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 3 \
#--adv_model AgNet --attacks ifgsm --target 2
