#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model SegNet --classes 3 \
#--attacks semantic --data_type org --device 0 --channels 1
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 3 \
#--attacks semantic --data_type org --device 0 --channels 1
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path octafull --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 3 \
#--attacks semantic --data_type org --device 0 --channels 1


#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 \
#--attacks dag --data_type DAG_A --device 0 --channels 1 --target 0 --mask_type 1 --suffix pvt_dag_cat_leff_3cpn
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 \
#--attacks dag --data_type DAG_A --device 0 --channels 1 --target 0 --mask_type 1 --suffix pvt_dag_cat_leff_3cpn
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 \
#--attacks dag --data_type DAG_A --device 0 --channels 1 --target 0 --mask_type 1 --suffix pvt_dag_cat_leff_3cpn
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --channels 1 --classes 2 --attacks dag --data_type DAG_A --mask_type 1 --target 0 --suffix pvt_dag_cat_leff_3cpn

#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 \
#--attacks dag --data_type DAG_D --device 0 --channels 1 --target 0 --mask_type 4 --suffix pvt_dag_cat_leff_3cpn
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 \
--attacks dag --data_type DAG_D --device 0 --channels 1 --target 0 --mask_type 4 --suffix pvt_dag_cat_leff_3cpn
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 \
--attacks dag --data_type DAG_D --device 0 --channels 1 --target 0 --mask_type 4 --suffix pvt_dag_cat_leff_3cpn
python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
 --channels 1 --classes 2 --attacks dag --data_type DAG_D --mask_type 4 --target 0 --suffix pvt_dag_cat_leff_3cpn

### for org images###
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model AgNet --classes 2 \
#--attacks dag --data_type org --device 0 --channels 1 --target 0 --suffix pvt_dag_cat_leff_3cpn
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model UNet --classes 2 \
#--attacks dag --data_type org --device 0 --channels 1 --target 0 --suffix pvt_dag_cat_leff_3cpn
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/ --model DenseNet --classes 2 \
#--attacks dag --data_type org --device 0 --channels 1 --target 0 --suffix pvt_dag_cat_leff_3cpn
#python tools/defense.py --model_path ./checkpoints/ --mode test --data_path lung --output_path \
#./output/test/denoise/ --denoise_output ./output/test/denoised_imgs/  --model deeplabv3plus_resnet101 --output_stride 8\
# --channels 1 --classes 2 --attacks dag --data_type org --target 0 --suffix pvt_dag_cat_leff_3cpn