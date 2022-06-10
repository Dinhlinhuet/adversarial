#python tools/test.py --model SegNet --channels 1 --classes 3 --output_path ./output/test/ --mode test --data_path octafull
#python tools/test.py --model UNet --channels 1 --classes 3  --output_path ./output/test/ --mode test --data_path octafull
#python tools/test.py --model DenseNet --channels 1 --classes 3  --output_path ./output/test/ --mode test --data_path octafull

#python tools/test_deeplab.py --model deeplabv3plus_mobilenet --output_stride 16 --channels 3 --classes 3 \
#--output_path ./output/test/ --data_path fundus

#python tools/test_deeplab.py --model deeplabv3plus_resnet101 --output_stride 8 --channels 3 --classes 3 \
#--output_path ./output/test/ --data_path fundus

#python tools/test_deeplab.py --model deeplabv3plus_mobilenet --output_stride 8 --channels 3 --classes 2 \
#--output_path ./output/test/ --data_path brain

#python tools/test_deeplab.py --model deeplabv3plus_resnet101 --output_stride 8 --channels 1 --classes 3 \
#--output_path ./output/test/ --data_path octafull

### test attack ###
#python tools/test.py --model SegNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_A --adv_model SegNet --mask_type 1 --data_path octafull
#python tools/test.py --model SegNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_B --adv_model SegNet --mask_type 2 --data_path octafull
#python tools/test.py --model SegNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_C --adv_model SegNet --mask_type 3 --target 1 --data_path octafull
#
#python tools/test.py --model SegNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_A --adv_model UNet --mask_type 1 --data_path octafull
#python tools/test.py --model SegNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_B --adv_model UNet --mask_type 2 --data_path octafull
#python tools/test.py --model SegNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_C --adv_model UNet --mask_type 3 --target 1 --data_path octafull
#
#python tools/test.py --model SegNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_A --adv_model DenseNet --mask_type 1 --data_path octafull
#python tools/test.py --model SegNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_B --adv_model DenseNet --mask_type 2 --data_path octafull
#python tools/test.py --model SegNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_C --adv_model DenseNet --mask_type 3 --target 1 --data_path octafull

#python tools/test.py --model UNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_A --adv_model SegNet --mask_type 1 --data_path octafull
#python tools/test.py --model UNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_B --adv_model SegNet --mask_type 2 --data_path octafull
#python tools/test.py --model UNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_C --adv_model SegNet --mask_type 3 --target 1 --data_path octafull
#
##python tools/test.py --model UNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
##./output/test/ --mode test --attacks DAG_A --adv_model UNet --mask_type 1 --data_path octafull
##python tools/test.py --model UNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
##./output/test/ --mode test --attacks DAG_B --adv_model UNet --mask_type 2 --data_path octafull
##python tools/test.py --model UNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
##./output/test/ --mode test --attacks DAG_C --adv_model UNet --mask_type 3 --target 1 --data_path octafull
#
#python tools/test.py --model UNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_A --adv_model DenseNet --mask_type 1 --data_path octafull
#python tools/test.py --model UNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_B --adv_model DenseNet --mask_type 2 --data_path octafull
#python tools/test.py --model UNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_C --adv_model DenseNet --mask_type 3 --target 1 --data_path octafull

#python tools/test.py --model DenseNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_A --adv_model SegNet --mask_type 1 --data_path octafull
#python tools/test.py --model DenseNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_B --adv_model SegNet --mask_type 2 --data_path octafull
#python tools/test.py --model DenseNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_C --adv_model SegNet --mask_type 3 --target 1 --data_path octafull
#
#python tools/test.py --model DenseNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_A --adv_model UNet --mask_type 1 --data_path octafull
#python tools/test.py --model DenseNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_B --adv_model UNet --mask_type 2 --data_path octafull
#python tools/test.py --model DenseNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_C --adv_model UNet --mask_type 3 --target 1 --data_path octafull
#
#python tools/test.py --model DenseNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_A --adv_model DenseNet --mask_type 1 --data_path octafull
#python tools/test.py --model DenseNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_B --adv_model DenseNet --mask_type 2 --data_path octafull
#python tools/test.py --model DenseNet --channels 1 --classes 3 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --attacks DAG_C --adv_model DenseNet --mask_type 3 --target 1 --data_path octafull


#  python tools/test.py --model UNet --channels 3 --model_path ./checkpoints/brain/UNet.pth --output_path \
#  ./output/test/ --mode test --attacks DAG_A --adv_model UNet --data_path brain
#python tools/test_ag.py --best_model ./checkpoints/fundus/AgNet.pth --mode test --data_path fundus --output_path \
#./output/test/ --model AgNet --adv_model UNet --attacks ifgsm --target 1

#python tools/test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model SegNet --classes 3 --adv_model SegNet --attacks ifgsm --target 1 --mask_type 1

#python tools/test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model UNet --classes 3 --adv_model UNet --attacks ifgsm --target 1 --mask_type 1

#python tools/test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model DenseNet --classes 3 --adv_model DenseNet --attacks ifgsm --target 1 --mask_type 1
#python tools/test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model DenseNet --classes 3 --adv_model DenseNet --attacks spt --target 0 --mask_type 2

#python tools/test_ag.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model AgNet --classes 3 --adv_model AgNet --attacks ifgsm --target 1 --mask_type 1 \
#--batch_size 2
#python tools/test_ag.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model AgNet --classes 3 --adv_model AgNet --attacks ifgsm --target r --mask_type 2 \
#--batch_size 2

#python tools/test_deeplab.py --output_stride 16 --channels 3 --classes 3 --data_path fundus --output_path \
#./output/test/ --model deeplabv3plus_resnet101 --adv_model deeplabv3plus_resnet101 \
#--attacks ifgsm --target 1 --mask_type 1 --batch_size 2
#python tools/test_deeplab.py --output_stride 16 --channels 3 --classes 3 --data_path fundus --output_path \
#./output/test/ --model deeplabv3plus_resnet101 --adv_model deeplabv3plus_resnet101 \
#--attacks spt --target 0 --mask_type 2 --batch_size 4



#python tools/test.py --model SegNet --channels 1 --classes 2 --output_path ./output/test/ --mode test --data_path lung
#python tools/test_ag.py --model AgNet --channels 1 --classes 2 --output_path ./output/test/ --mode test --data_path lung\
# --data_type org --batch_size 4
#python tools/test.py --model UNet --channels 1 --classes 2  --output_path ./output/test/ --mode test --data_path lung \
#--device 1
#python tools/test.py --model DenseNet --channels 1 --classes 2  --output_path ./output/test/ --mode test --data_path lung
#python tools/test_deeplab.py --output_stride 16 --channels 1 --classes 2 --data_path lung --output_path \
#./output/test/ --model deeplabv3plus_resnet101 --adv_model deeplabv3plus_resnet101 \
#--batch_size 4


##python tools/test.py --model UNet --channels 1 --classes 2 --model_path ./checkpoints/ --output_path \
##./output/test/ --mode test --data_type DAG_A --adv_model UNet --mask_type 1 --data_path lung --attacks dag
#python tools/test.py --model UNet --channels 1 --classes 2 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --data_type DAG_D --adv_model UNet --mask_type 4 --target 0 --data_path lung --attacks dag \
#--device 1

python tools/test_ag.py --model AgNet --channels 1 --classes 2 --model_path ./checkpoints/ --output_path \
./output/test/ --mode test --data_type DAG_A --adv_model AgNet --mask_type 1 --data_path lung --attacks dag --device 1
python tools/test_ag.py --model AgNet --channels 1 --classes 2 --model_path ./checkpoints/ --output_path \
./output/test/ --mode test --data_type DAG_D --adv_model AgNet --mask_type 4 --target 0 --data_path lung --attacks dag \
--device 1 --batch_size 4

##python tools/test.py --model DenseNet --channels 1 --classes 2 --model_path ./checkpoints/ --output_path \
##./output/test/ --mode test --data_type DAG_A --adv_model DenseNet --mask_type 1 --data_path lung --attacks dag
#python tools/test.py --model DenseNet --channels 1 --classes 2 --model_path ./checkpoints/ --output_path \
#./output/test/ --mode test --data_type DAG_D --adv_model DenseNet --mask_type 4 --target 0 --data_path lung --attacks dag \
#--device 1

##python tools/test_deeplab.py --model deeplabv3plus_resnet101 --channels 1 --classes 2 --output_path \
##./output/test/ --mode test --data_type DAG_A --adv_model deeplabv3plus_resnet101 --mask_type 1 --data_path lung \
##--attacks dag
#python tools/test_deeplab.py --model deeplabv3plus_resnet101 --channels 1 --classes 2  --output_path \
#./output/test/ --mode test --data_type DAG_D --adv_model deeplabv3plus_resnet101 --mask_type 4 --target 0 --data_path lung \
# --attacks dag --device 1