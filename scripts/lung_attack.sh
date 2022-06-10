#python tools/adversarials.py --attacks DAG_A --target 0 --mask_type 1 --data_path lung \
#--model_path ./checkpoints/ --model AgNet --classes 2 --channels 1 --mode test --batch_size 8
#python tools/adversarials.py --attacks DAG_B --target 0 --mask_type 2 --data_path lung \
#--model_path ./checkpoints/ --model AgNet --classes 2 --channels 1 --mode test --batch_size 8
#python tools/adversarials.py --attacks DAG_C --target 1 --mask_type 3 --data_path lung \
#--model_path ./checkpoints/ --model AgNet --classes 2 --channels 1 --mode test --batch_size 8

#python tools/adversarials.py --data_type DAG_A --target 0 --mask_type 1 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model AgNet --classes 2 --channels 1 --mode test --batch_size 4 --device 1
#python tools/adversarials.py --data_type DAG_D --target 0 --mask_type 4 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model AgNet --classes 2 --channels 1 --mode test --batch_size 4 --device 1

#python tools/adversarials.py --data_type DAG_A --target 0 --mask_type 1 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model UNet --classes 2 --channels 1 --mode test --batch_size 8 --device 1
#python tools/adversarials.py --data_type DAG_D --target 0 --mask_type 4 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model UNet --classes 2 --channels 1 --mode test --batch_size 8 --device 1

#python tools/adversarials.py --data_type DAG_A --target 0 --mask_type 1 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model DenseNet --classes 2 --channels 1 --mode test --batch_size 4 --device 1
#python tools/adversarials.py --data_type DAG_D --target 0 --mask_type 4 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model DenseNet --classes 2 --channels 1 --mode test --batch_size 4 --device 0

#python tools/adversarials.py --data_type DAG_A --target 0 --mask_type 1 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model deeplabv3plus_resnet101 --classes 2 --channels 1 --mode test --batch_size 8 --device 1
#python tools/adversarials.py --data_type DAG_D --target 0 --mask_type 4 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model deeplabv3plus_resnet101 --classes 2 --channels 1 --mode test --batch_size 8 --device 1

#python tools/adversarials.py --attacks DAG_A --target 0 --mask_type 1 --data_path lung \
#--model_path ./checkpoints/ --model deeplabv3plus_resnet101 --classes 2 --channels 1 --mode train --batch_size 8
#python tools/adversarials.py --attacks DAG_B --target 0 --mask_type 2 --data_path lung \
#--model_path ./checkpoints/ --model deeplabv3plus_resnet101 --classes 2 --channels 1 --mode train --batch_size 8
#python tools/adversarials.py --attacks DAG_C --target 1 --mask_type 3 --data_path lung \
#--model_path ./checkpoints/ --model deeplabv3plus_resnet101 --classes 2 --channels 1 --mode train --batch_size 8


###ifgsm###
#python tools/adversarials.py --target 0 --mask_type 1 --data_path lung --attacks ifgsm \
#--model_path ./checkpoints/ --model AgNet --classes 2 --channels 1 --mode test --batch_size 2 --device 1

#python tools/adversarials.py --target 0 --mask_type 1 --data_path lung --attacks ifgsm \
#--model_path ./checkpoints/ --model UNet --classes 2 --channels 1 --mode test --batch_size 8 --device 1
##python tools/adversarials.py --data_type DAG_E --target 0 --mask_type 5 --data_path lung --attacks ifgsm \
##--model_path ./checkpoints/ --model UNet --classes 2 --channels 1 --mode test --batch_size 8 --device 1
#
#python tools/adversarials.py --target 0 --mask_type 1 --data_path lung --attacks ifgsm \
#--model_path ./checkpoints/ --model DenseNet --classes 2 --channels 1 --mode test --batch_size 4 --device 1
##python tools/adversarials.py --data_type DAG_E --target 0 --mask_type 5 --data_path lung --attacks ifgsm \
##--model_path ./checkpoints/ --model DenseNet --classes 2 --channels 1 --mode test --batch_size 4 --device 1
#
#python tools/adversarials.py --target 0 --mask_type 1 --data_path lung --attacks ifgsm \
#--model_path ./checkpoints/ --model deeplabv3plus_resnet101 --classes 2 --channels 1 --mode test --batch_size 8 --device 1
##python tools/adversarials.py --data_type DAG_E --target 0 --mask_type 5 --data_path lung --attacks ifgsm \
##--model_path ./checkpoints/ --model deeplabv3plus_resnet101 --classes 2 --channels 1 --mode test --batch_size 8 --device 1

###for training###
python tools/adversarials.py --target 0 --mask_type 1 --data_path lung --attacks ifgsm \
--model_path ./checkpoints/ --model AgNet --classes 2 --channels 1 --mode train --batch_size 8 --device 0

#python tools/adversarials.py --target 0 --mask_type 1 --data_path lung --attacks ifgsm \
#--model_path ./checkpoints/ --model UNet --classes 2 --channels 1 --mode train --batch_size 8 --device 0
#
#python tools/adversarials.py --target 0 --mask_type 1 --data_path lung --attacks ifgsm \
#--model_path ./checkpoints/ --model DenseNet --classes 2 --channels 1 --mode train --batch_size 2 --device 0
##
#python tools/adversarials.py --target 0 --mask_type 1 --data_path lung --attacks ifgsm \
#--model_path ./checkpoints/ --model deeplabv3plus_resnet101 --classes 2 --channels 1 --mode train --batch_size 4 --device 0

##for validation
#python tools/adversarials.py --target 0 --mask_type 1 --data_path lung --attacks ifgsm \
#--model_path ./checkpoints/ --model AgNet --classes 2 --channels 1 --mode val --batch_size 4 --device 1

#python tools/adversarials.py --target 0 --mask_type 1 --data_path lung --attacks ifgsm \
#--model_path ./checkpoints/ --model UNet --classes 2 --channels 1 --mode val --batch_size 8 --device 1
#
#python tools/adversarials.py --target 0 --mask_type 1 --data_path lung --attacks ifgsm \
#--model_path ./checkpoints/ --model DenseNet --classes 2 --channels 1 --mode val --batch_size 2 --device 1
#
#python tools/adversarials.py --target 0 --mask_type 1 --data_path lung --attacks ifgsm \
#--model_path ./checkpoints/ --model deeplabv3plus_resnet101 --classes 2 --channels 1 --mode val --batch_size 4 --device 1

##for training##
#python tools/adversarials.py --data_type DAG_A --target 0 --mask_type 1 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model AgNet --classes 2 --channels 1 --mode train --batch_size 2 --device 0
#python tools/adversarials.py --data_type DAG_D --target 0 --mask_type 4 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model AgNet --classes 2 --channels 1 --mode train --batch_size 2 --device 0

#python tools/adversarials.py --data_type DAG_A --target 0 --mask_type 1 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model UNet --classes 2 --channels 1 --mode train --batch_size 8 --device 0
#python tools/adversarials.py --data_type DAG_D --target 0 --mask_type 4 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model UNet --classes 2 --channels 1 --mode train --batch_size 8 --device 0
#
python tools/adversarials.py --data_type DAG_A --target 0 --mask_type 1 --data_path lung --attacks dag \
--model_path ./checkpoints/ --model DenseNet --classes 2 --channels 1 --mode train --batch_size 4 --device 0
#python tools/adversarials.py --data_type DAG_D --target 0 --mask_type 4 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model DenseNet --classes 2 --channels 1 --mode train --batch_size 4 --device 0
#
#python tools/adversarials.py --data_type DAG_A --target 0 --mask_type 1 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model deeplabv3plus_resnet101 --classes 2 --channels 1 --mode train --batch_size 8 --device 0
#python tools/adversarials.py --data_type DAG_D --target 0 --mask_type 4 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model deeplabv3plus_resnet101 --classes 2 --channels 1 --mode train --batch_size 8 --device 0

### for validation###
#python tools/adversarials.py --data_type DAG_A --target 0 --mask_type 1 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model AgNet --classes 2 --channels 1 --mode val --batch_size 2 --device 0
#python tools/adversarials.py --data_type DAG_D --target 0 --mask_type 4 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model AgNet --classes 2 --channels 1 --mode val --batch_size 2 --device 0
#
#python tools/adversarials.py --data_type DAG_A --target 0 --mask_type 1 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model UNet --classes 2 --channels 1 --mode val --batch_size 8 --device 0
#python tools/adversarials.py --data_type DAG_D --target 0 --mask_type 4 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model UNet --classes 2 --channels 1 --mode val --batch_size 8 --device 0
#
#python tools/adversarials.py --data_type DAG_A --target 0 --mask_type 1 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model DenseNet --classes 2 --channels 1 --mode val --batch_size 4 --device 0
#python tools/adversarials.py --data_type DAG_D --target 0 --mask_type 4 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model DenseNet --classes 2 --channels 1 --mode val --batch_size 4 --device 0
#
#python tools/adversarials.py --data_type DAG_A --target 0 --mask_type 1 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model deeplabv3plus_resnet101 --classes 2 --channels 1 --mode val --batch_size 8 --device 0
#python tools/adversarials.py --data_type DAG_D --target 0 --mask_type 4 --data_path lung --attacks dag \
#--model_path ./checkpoints/ --model deeplabv3plus_resnet101 --classes 2 --channels 1 --mode val --batch_size 8 --device 0