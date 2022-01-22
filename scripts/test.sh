#python tools/test.py --model SegNet --channels 1 --classes 3 --output_path ./output/test/ --mode test --data_path octafull
#python tools/test.py --model UNet --channels 1 --classes 3  --output_path ./output/test/ --mode test --data_path octafull
#python tools/test.py --model DenseNet --channels 1 --classes 3  --output_path ./output/test/ --mode test --data_path octafull

#python tools/test_deeplab.py --model deeplabv3plus_mobilenet --output_stride 16 --channels 3 --classes 3 \
#--output_path ./output/test/ --data_path fundus

python tools/test_deeplab.py --model deeplabv3plus_resnet101 --output_stride 8 --channels 3 --classes 3 \
--output_path ./output/test/ --data_path fundus

#python tools/test_deeplab.py --model deeplabv3plus_mobilenet --output_stride 8 --channels 3 --classes 2 \
#--output_path ./output/test/ --data_path brain

#  python test.py --model UNet --channels 3 --model_path ./checkpoints/brain/UNet.pth --output_path \
#  ./output/test/ --mode test --attacks DAG_A --adv_model UNet --data_path brain
#python test_ag.py --best_model ./checkpoints/fundus/AgNet.pth --mode test --data_path fundus --output_path \
#./output/test/ --model AgNet --adv_model UNet --attacks ifgsm --target 1

#python test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model SegNet --classes 3 --adv_model SegNet --attacks ifgsm --target 1 --mask_type 1
#python test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model SegNet --classes 3 --adv_model UNet --attacks ifgsm --target 1 --mask_type 1
#python test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model SegNet --classes 3 --adv_model DenseNet --attacks ifgsm --target 1 --mask_type 1
#python test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model SegNet --classes 3 --adv_model AgNet --attacks ifgsm --target 1 --mask_type 1

#python test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model UNet --classes 3 --adv_model SegNet --attacks ifgsm --target 1 --mask_type 1
#python test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model UNet --classes 3 --adv_model UNet --attacks ifgsm --target 1 --mask_type 1
#python test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model UNet --classes 3 --adv_model DenseNet --attacks ifgsm --target 1 --mask_type 1
#python test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model UNet --classes 3 --adv_model AgNet --attacks ifgsm --target 1 --mask_type 1

#python test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model DenseNet --classes 3 --adv_model SegNet --attacks ifgsm --target 1 --mask_type 1
#python test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model DenseNet --classes 3 --adv_model UNet --attacks ifgsm --target 1 --mask_type 1
#python test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model DenseNet --classes 3 --adv_model DenseNet --attacks ifgsm --target 1 --mask_type 1
#python test.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model DenseNet --classes 3 --adv_model AgNet --attacks ifgsm --target 1 --mask_type 1

#python test_ag.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model AgNet --classes 3 --adv_model SegNet --attacks ifgsm --target 1 --mask_type 1 \
#--batch_size 4
#python test_ag.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model AgNet --classes 3 --adv_model UNet --attacks ifgsm --target 1 --mask_type 1 \
#--batch_size 4
#python test_ag.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model AgNet --classes 3 --adv_model DenseNet --attacks ifgsm --target 1 --mask_type 1 \
#--batch_size 2
#python test_ag.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model AgNet --classes 3 --adv_model AgNet --attacks ifgsm --target 1 --mask_type 1 \
#--batch_size 2
