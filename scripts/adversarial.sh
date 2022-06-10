#python tools/adversarials.py --attacks DAG_A  --model_path ./checkpoints/UNet.pth --model Unet
#python tools/adversarials.py --attacks ifgsm --mask_type 1 --target 1 --data_path fundus --model_path ./checkpoints/ --model SegNet\
#  --classes 3 --batch_size 2 --mode train
#python tools/adversarials.py --attacks ifgsm --mask_type 2 --target r --data_path fundus --model_path ./checkpoints/ \
#--model SegNet --classes 3 --batch_size 4 --mode train
#python tools/adversarials.py --attacks ifgsm --mask_type 1 --target 1 --data_path fundus --model_path ./checkpoints/ --model deeplabv3plus_resnet101\
#  --classes 3 --batch_size 2 --mode train
#python tools/adversarials.py --attacks ifgsm --mask_type 2 --target r --data_path fundus --model_path ./checkpoints/ \
#--model deeplabv3plus_resnet101 --classes 3 --batch_size 4 --mode train

#python tools/adversarials.py --attacks ifgsm --mask_type 1 --target 1 --data_path fundus --model_path ./checkpoints/ --model SegNet\
#  --classes 3 --batch_size 2 --mode val
#python tools/adversarials.py --attacks ifgsm --mask_type 2 --target r --data_path fundus --model_path ./checkpoints/ \
#--model SegNet --classes 3 --batch_size 4 --mode val
#python tools/adversarials.py --attacks ifgsm --mask_type 1 --target 1 --data_path fundus --model_path ./checkpoints/ --model UNet\
#  --classes 3 --batch_size 2 --mode val
#python tools/adversarials.py --attacks ifgsm --mask_type 2 --target r --data_path fundus --model_path ./checkpoints/ \
#--model UNet --classes 3 --batch_size 4 --mode val
#python tools/adversarials.py --attacks ifgsm --mask_type 1 --target 1 --data_path fundus --model_path ./checkpoints/ --model DenseNet\
#  --classes 3 --batch_size 2 --mode val
#python tools/adversarials.py --attacks ifgsm --mask_type 2 --target r --data_path fundus --model_path ./checkpoints/ \
#--model DenseNet --classes 3 --batch_size 4 --mode val
#python tools/adversarials.py --attacks ifgsm --mask_type 1 --target 1 --data_path fundus --model_path ./checkpoints/ --model AgNet\
#  --classes 3 --batch_size 2 --mode val
#python tools/adversarials.py --attacks ifgsm --mask_type 2 --target r --data_path fundus --model_path ./checkpoints/ \
#--model AgNet --classes 3 --batch_size 4 --mode val
#python tools/adversarials.py --attacks ifgsm --mask_type 1 --target 1 --data_path fundus --model_path ./checkpoints/ --model deeplabv3plus_resnet101\
#  --classes 3 --batch_size 2 --mode val
#python tools/adversarials.py --attacks ifgsm --mask_type 2 --target r --data_path fundus --model_path ./checkpoints/ \
#--model deeplabv3plus_resnet101 --classes 3 --batch_size 4 --mode val

#python tools/adversarials.py --attacks ifgsm --mask_type 1 --target 1 --data_path fundus --model_path ./checkpoints/
#--model DenseNet --classes 3 --batch_size 4 --mode test
#python tools/adversarials.py --attacks ifgsm --mask_type 3 --target 1 --data_path fundus --model_path ./checkpoints/ \
#--model DenseNet --classes 3 --batch_size 4 --mode test
#python tools/adversarials.py --attacks ifgsm --mask_type 1 --target 1 --data_path fundus --model_path ./checkpoints/\
# --model AgNet --classes 3 --batch_size 2 --mode test
#python tools/adversarials.py --attacks ifgsm --mask_type 1 --target 1 --data_path fundus --model_path ./checkpoints/ \
#--model deeplabv3plus_resnet101 --classes 3 --batch_size 4 --mode test
#python tools/adversarials.py --attacks ifgsm --mask_type 3 --target 1 --data_path fundus --model_path ./checkpoints/ \
#--model deeplabv3plus_resnet101 --classes 3 --batch_size 4 --mode test

#python tools/adversarials.py --attacks DAG_A --target 0 --mask_type 1 --data_path octafull \
#--model_path ./checkpoints/ --model SegNet --classes 3 --channels 1 --mode test --batch_size 8
#python tools/adversarials.py --attacks DAG_B --target 0 --mask_type 2 --data_path octafull \
#--model_path ./checkpoints/ --model SegNet --classes 3 --channels 1 --mode test --batch_size 8
#python tools/adversarials.py --attacks DAG_C --target 1 --mask_type 3 --data_path octafull \
#--model_path ./checkpoints/ --model SegNet --classes 3 --channels 1 --mode test --batch_size 8

#python tools/adversarials.py --attacks DAG_A --target 0 --mask_type 1 --data_path octafull \
#--model_path ./checkpoints/ --model UNet --classes 3 --channels 1 --mode test --batch_size 8
#python tools/adversarials.py --attacks DAG_B --target 0 --mask_type 2 --data_path octafull \
#--model_path ./checkpoints/ --model UNet --classes 3 --channels 1 --mode test --batch_size 8
#python tools/adversarials.py --attacks DAG_C --target 1 --mask_type 3 --data_path octafull \
#--model_path ./checkpoints/ --model UNet --classes 3 --channels 1 --mode test --batch_size 8

#python tools/adversarials.py --attacks DAG_A --target 0 --mask_type 1 --data_path octafull \
#--model_path ./checkpoints/ --model DenseNet --classes 3 --channels 1 --mode test --batch_size 4
#python tools/adversarials.py --attacks DAG_B --target 0 --mask_type 2 --data_path octafull \
#--model_path ./checkpoints/ --model DenseNet --classes 3 --channels 1 --mode test --batch_size 4
#python tools/adversarials.py --attacks DAG_C --target 1 --mask_type 3 --data_path octafull \
#--model_path ./checkpoints/ --model DenseNet --classes 3 --channels 1 --mode test --batch_size 4

#python tools/adversarials.py --attacks DAG_A --target 0 --mask_type 1 --data_path octafull \
#--model_path ./checkpoints/ --model SegNet --classes 3 --channels 1 --mode train --batch_size 8
#python tools/adversarials.py --attacks DAG_E --target 0 --mask_type 5 --data_path octafull \
#--model_path ./checkpoints/ --model SegNet --classes 3 --channels 1 --mode train --batch_size 8
#
##python tools/adversarials.py --attacks DAG_A --target 0 --mask_type 1 --data_path octafull \
##--model_path ./checkpoints/ --model UNet --classes 3 --channels 1 --mode train --batch_size 8
#python tools/adversarials.py --attacks DAG_E --target 0 --mask_type 5 --data_path octafull \
#--model_path ./checkpoints/ --model UNet --classes 3 --channels 1 --mode train --batch_size 8
#
##python tools/adversarials.py --attacks DAG_A --target 0 --mask_type 1 --data_path octafull \
##--model_path ./checkpoints/ --model DenseNet --classes 3 --channels 1 --mode train --batch_size 2
#python tools/adversarials.py --attacks DAG_E --target 0 --mask_type 5 --data_path octafull \
#--model_path ./checkpoints/ --model DenseNet --classes 3 --channels 1 --mode train --batch_size 2

#python tools/adversarials.py --attacks DAG_A  --mask_type 1 --target 0 --data_path brain --model_path ./checkpoints/ \
#--model SegNet --classes 2 --channels 3 --mode test --batch_size 4
#python tools/adversarials.py --attacks DAG_D  --mask_type 4 --target 1 --data_path brain --model_path ./checkpoints/ \
#--model SegNet --classes 2 --channels 3 --mode test --batch_size 4

#python tools/adversarials.py --attacks DAG_A  --mask_type 1 --target 0 --data_path brain --model_path ./checkpoints/
#--model UNet --classes 2 --channels 3 --mode test --batch_size 4
#python tools/adversarials.py --attacks DAG_D  --mask_type 4 --target 1 --data_path brain --model_path ./checkpoints/ \
#--model UNet --classes 2 --channels 3 --mode test --batch_size 4

#python tools/adversarials.py --attacks DAG_A  --mask_type 1 --target 0 --data_path brain --model_path ./checkpoints/ \
#--model DenseNet --classes 2 --channels 3 --mode test --batch_size 2
#python tools/adversarials.py --attacks DAG_D  --mask_type 4 --target 1 --data_path brain --model_path ./checkpoints/ \
#--model DenseNet --classes 2 --channels 3 --mode test --batch_size 2

#python tools/adversarials.py --attacks spt --mask_type 2 --target 0 --data_path fundus --model_path ./checkpoints/ \
#--model DenseNet --classes 3 --batch_size 4 --mode test
#python tools/adversarials.py --attacks spt --mask_type 2 --target 0 --data_path fundus --model_path ./checkpoints/ \
#--model deeplabv3plus_resnet101 --classes 3 --batch_size 4 --mode test --device 1

#python tools/adversarials.py --attacks square --mask_type 2 --target 0 --data_path fundus --model_path ./checkpoints/ \
#--model DenseNet --classes 3 --batch_size 4 --mode test


#python tools/adversarials.py --data_type DAG_A  --mask_type 1 --target 0 --data_path octafull --model_path ./checkpoints/ \
#--model SegNet --classes 3 --channels 1 --mode test --batch_size 4 --attacks ifgsm
#python tools/adversarials.py --data_type DAG_C  --mask_type 3 --target 1 --data_path octafull --model_path ./checkpoints/ \
#--model SegNet --classes 3 --channels 1 --mode test --batch_size 4 --attacks ifgsm
#
#python tools/adversarials.py --data_type DAG_A  --mask_type 1 --target 0 --data_path octafull --model_path ./checkpoints/
#--model UNet --classes 3 --channels 1 --mode test --batch_size 4 --attacks ifgsm
#python tools/adversarials.py --data_type DAG_C  --mask_type 3 --target 1 --data_path octafull --model_path ./checkpoints/ \
#--model UNet --classes 3 --channels 1 --mode test --batch_size 4 --attacks ifgsm
#
#python tools/adversarials.py --data_type DAG_A  --mask_type 1 --target 0 --data_path octafull --model_path ./checkpoints/ \
#--model DenseNet --classes 3 --channels 1 --mode test --batch_size 2 --attacks ifgsm
#python tools/adversarials.py --data_type DAG_C  --mask_type 3 --target 1 --data_path octafull --model_path ./checkpoints/ \
#--model DenseNet --classes 3 --channels 1 --mode test --batch_size 2 --attacks ifgsm


python tools/adversarials.py --data_type DAG_A  --mask_type 1 --target 0 --data_path octafull --model_path ./checkpoints/ \
--model SegNet --classes 3 --channels 1 --mode train --batch_size 4 --attacks ifgsm --device 1
python tools/adversarials.py --data_type DAG_C  --mask_type 3 --target 1 --data_path octafull --model_path ./checkpoints/ \
--model SegNet --classes 3 --channels 1 --mode train --batch_size 4 --attacks ifgsm --device 1

python tools/adversarials.py --data_type DAG_A  --mask_type 1 --target 0 --data_path octafull --model_path ./checkpoints/ \
--model UNet --classes 3 --channels 1 --mode train --batch_size 4 --attacks ifgsm --device 1
python tools/adversarials.py --data_type DAG_C  --mask_type 3 --target 1 --data_path octafull --model_path ./checkpoints/ \
--model UNet --classes 3 --channels 1 --mode train --batch_size 4 --attacks ifgsm --device 1

python tools/adversarials.py --data_type DAG_A  --mask_type 1 --target 0 --data_path octafull --model_path ./checkpoints/ \
--model DenseNet --classes 3 --channels 1 --mode train --batch_size 2 --attacks ifgsm --device 1
python tools/adversarials.py --data_type DAG_C  --mask_type 3 --target 1 --data_path octafull --model_path ./checkpoints/ \
--model DenseNet --classes 3 --channels 1 --mode train --batch_size 2 --attacks ifgsm --device 1