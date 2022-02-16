#python test.py --model SegNet --channels 3 --model_path ./checkpoints/lung/SegNet.pth --output_path
# ./output/test/ --mode test --attacks DAG_C --adv_model SegNet --data_path lung
#  python test.py --model UNet --channels 3 --model_path ./checkpoints/brain/UNet.pth --output_path \
#  ./output/test/ --mode test --attacks DAG_A --adv_model UNet --data_path brain
#python test_ag.py --best_model ./checkpoints/fundus/AgNet.pth --mode test --data_path fundus --output_path \
#./output/test/ --model AgNet --adv_model UNet --attacks ifgsm --target 1


#python tools/test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model SegNet --classes 3 --adv_model SegNet --attacks semantic --target 0 --mask_type 5

#python tools/test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model UNet --classes 3 --adv_model UNet --attacks semantic --target 0 --mask_type 5
#
#python tools/test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model DenseNet --classes 3 --adv_model DenseNet --attacks semantic --target 0 --mask_type 5

#python tools/test_ag.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model AgNet --classes 3 --adv_model AgNet --attacks semantic --target 0 --mask_type 5 \
##--batch_size 2

#python tools/test_deeplab.py --output_stride 16 --channels 3 --classes 3 --data_path fundus --output_path \
#./output/test/ --model deeplabv3plus_resnet101 --adv_model deeplabv3plus_resnet101 \
#--attacks semantic --target 0 --mask_type 5 --batch_size 4


#python tools/test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model SegNet --classes 3 --adv_model SegNet --attacks semantic --suffix mix_label
#
##python tools/test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
##./output/test/ --model SegNet --classes 3 --adv_model SegNet --attacks semantic --target 0 --mask_type 2
#
#python tools/test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model UNet --classes 3 --adv_model UNet --attacks semantic --suffix mix_label
#
#python tools/test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model DenseNet --classes 3 --adv_model DenseNet --attacks semantic --suffix mix_label
#
#python tools/test_ag.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model AgNet --classes 3 --adv_model AgNet --attacks semantic --suffix mix_label \
#--batch_size 2
#
#python tools/test_deeplab.py --output_stride 16 --channels 3 --classes 3 --data_path fundus --output_path \
#./output/test/ --model deeplabv3plus_resnet101 --adv_model deeplabv3plus_resnet101 \
#--attacks semantic --suffix mix_label --batch_size 2

#python tools/test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model SegNet --classes 3 --adv_model SegNet --attacks semantic --suffix pure_target
#
#python tools/test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model UNet --classes 3 --adv_model UNet --attacks semantic --suffix pure_target
#
#python tools/test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model DenseNet --classes 3 --adv_model DenseNet --attacks semantic --suffix pure_target
#
#python tools/test_ag.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model AgNet --classes 3 --adv_model AgNet --attacks semantic --suffix pure_target \
#--batch_size 2
#
#python tools/test_deeplab.py --output_stride 16 --channels 3 --classes 3 --data_path fundus --output_path \
#./output/test/ --model deeplabv3plus_resnet101 --adv_model deeplabv3plus_resnet101 \
#--attacks semantic --suffix pure_target --batch_size 2

##brain##
#python tools/test.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/ --model SegNet --classes 2 --adv_model SegNet --attacks semantic --suffix mix_label
#
#python tools/test.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/ --model UNet --classes 2 --adv_model UNet --attacks semantic --suffix mix_label
#
#python tools/test.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
#./output/test/ --model DenseNet --classes 2 --adv_model DenseNet --attacks semantic --suffix mix_label
#
#python tools/test_deeplab.py --output_stride 16 --channels 3 --classes 2 --data_path brain --output_path \
#./output/test/ --model deeplabv3plus_mobilenet --adv_model deeplabv3plus_mobilenet \
#--attacks semantic --suffix mix_label --batch_size 2


python tools/test.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
./output/test/ --model SegNet --classes 2 --adv_model SegNet --attacks semantic --target 0 --mask_type 5 --device 1

python tools/test.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
./output/test/ --model UNet --classes 2 --adv_model UNet --attacks semantic --target 0 --mask_type 5 --device 1

python tools/test.py --model_path ./checkpoints/ --mode test --data_path brain --output_path \
./output/test/ --model DenseNet --classes 2 --adv_model DenseNet --attacks semantic --target 0 --mask_type 5 --device 1

python tools/test_deeplab.py --output_stride 16 --channels 3 --classes 2 --data_path brain --output_path \
./output/test/ --model deeplabv3plus_mobilenet --adv_model deeplabv3plus_mobilenet \
--attacks semantic --target 0 --mask_type 5 --batch_size 2 --device 1
