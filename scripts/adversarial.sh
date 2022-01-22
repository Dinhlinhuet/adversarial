#python adversarials.py --attacks DAG_A  --model_path ./checkpoints/UNet.pth --model Unet
#python adversarials.py --attacks ifgsm --mask_type 1 --data_path fundus --model_path ./checkpoints/ --model DenseNet\
#  --classes 3 --batch_size 4 --mode test
#python adversarials.py --attacks ifgsm --mask_type 1 --data_path fundus --model_path ./checkpoints/ --model AgNet\
#  --classes 3 --batch_size 2 --mode test
#python adversarials.py --attacks ifgsm --mask_type 1 --target 2 --data_path fundus --model_path ./checkpoints/ --model DenseNet\
#  --classes 3 --batch_size 4 --mode test
#python adversarials.py --attacks DAG_A --target 0 --data_path octa3mfull --model_path ./checkpoints/ --model UNet\
#  --classes 3 --channels 1 --mode train
#python adversarials.py --attacks DAG_A --target 0 --data_path octa3mfull --model_path ./checkpoints/ --model UNet\
#  --classes 3 --channels 1 --mode test --batch_size 4
#python adversarials.py --attacks DAG_C --target 1 --data_path octa3mfull --model_path ./checkpoints/ --model UNet\
#  --classes 3 --channels 1 --mode test --batch_size 4
#python adversarials.py --attacks cw  --mask_type 1 --target 1 --data_path octa3mfull --model_path ./checkpoints/ \
#--model SegNet --classes 3 --channels 1 --mode test --batch_size 4
python adversarials.py --attacks cw  --mask_type 3 --target 0 --data_path octa3mfull --model_path ./checkpoints/ \
--model SegNet --classes 3 --channels 1 --mode test --batch_size 4
#python adversarials.py --attacks cw  --mask_type 3 --target 1 --data_path octa3mfull --model_path ./checkpoints/ \
#--model SegNet --classes 3 --channels 1 --mode test --batch_size 4
#python adversarials.py --attacks cw  --mask_type 3 --target 2 --data_path octa3mfull --model_path ./checkpoints/ \
#--model SegNet --classes 3 --channels 1 --mode test --batch_size 4

#python adversarials.py --attacks cw  --mask_type 1 --target 1 --data_path octa3mfull --model_path ./checkpoints/ --model UNet \
#--classes 3 --channels 1 --mode test --batch_size 4
python adversarials.py --attacks cw  --mask_type 3 --target 0 --data_path octa3mfull --model_path ./checkpoints/ --model UNet \
--classes 3 --channels 1 --mode test --batch_size 4
#python adversarials.py --attacks cw  --mask_type 3 --target 1 --data_path octa3mfull --model_path ./checkpoints/ --model UNet \
#--classes 3 --channels 1 --mode test --batch_size 4
#python adversarials.py --attacks cw  --mask_type 3 --target 2 --data_path octa3mfull --model_path ./checkpoints/ --model UNet \
#--classes 3 --channels 1 --mode test --batch_size 4

#python adversarials.py --attacks cw  --mask_type 1 --target 1 --data_path octa3mfull --model_path ./checkpoints/ \
#--model DenseNet --classes 3 --channels 1 --mode test --batch_size 2
python adversarials.py --attacks cw  --mask_type 3 --target 0 --data_path octa3mfull --model_path ./checkpoints/ \
--model DenseNet --classes 3 --channels 1 --mode test --batch_size 2
#python adversarials.py --attacks cw  --mask_type 3 --target 1 --data_path octa3mfull --model_path ./checkpoints/ \
#--model DenseNet --classes 3 --channels 1 --mode test --batch_size 2
#python adversarials.py --attacks cw  --mask_type 3 --target 2 --data_path octa3mfull --model_path ./checkpoints/ \
#--model DenseNet --classes 3 --channels 1 --mode test --batch_size 2