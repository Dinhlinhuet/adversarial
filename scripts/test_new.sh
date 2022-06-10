#python test_ag.py --model_path ./checkpoints/ --mode test --output_path \
#./output/test/ --model AgNet --batch_size 4 --classes 3 --data_path fundus --suffix scl_attk
#python tools/test.py --model_path ./checkpoints/ --mode test --output_path \
#./output/test/ --model SegNet --batch_size 4 --classes 3 --data_path fundus --suffix scl_attk
#python tools/test.py --model_path ./checkpoints/ --mode test --output_path \
#./output/test/ --model UNet --batch_size 4 --classes 3 --data_path fundus --suffix scl_attk
#python tools/test.py --model_path ./checkpoints/ --mode test --output_path \
#./output/test/ --model DenseNet --batch_size 4 --classes 3 --data_path fundus --suffix scl_attk

#python test_ag.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model AgNet --classes 3 --data_type org --suffix retest

#python test_ag.py --model_path ./checkpoints/ --mode test --output_path \
#./output/test/ --model AgNet --batch_size 4 --classes 2 --data_path brain --suffix scl_attk
#python tools/test.py --model_path ./checkpoints/ --mode test --output_path \
#./output/test/ --model SegNet --batch_size 4 --classes 2 --data_path brain --suffix scl_attk
#python tools/test.py --model_path ./checkpoints/ --mode test --output_path \
#./output/test/ --model UNet --batch_size 4 --classes 2 --data_path brain --suffix scl_attk
#python tools/test.py --model_path ./checkpoints/ --mode test --output_path \
#./output/test/ --model DenseNet --batch_size 4 --classes 2 --data_path brain --suffix scl_attk

#python tools/test.py --model_path ./checkpoints/ --mode test --channels 1 --output_path \
#./output/test/ --model SegNet --batch_size 4 --classes 3 --data_path octafull --suffix scl_attk
#python tools/test.py --model_path ./checkpoints/ --mode test --channels 1  --output_path \
#./output/test/ --model UNet --batch_size 4 --classes 3 --data_path octafull --suffix scl_attk
#python tools/test.py --model_path ./checkpoints/ --mode test --channels 1  --output_path \
#./output/test/ --model DenseNet --batch_size 4 --classes 3 --data_path octafull --suffix scl_attk

#python tools/test_ag.py --model_path ./checkpoints/ --mode test --output_path \
#./output/test/ --model AgNet --batch_size 4 --classes 2 --channels 1  --data_path lung --attacks scl_attk
#python tools/test.py --model_path ./checkpoints/ --mode test --output_path \
#./output/test/ --model UNet --batch_size 4 --classes 2 --channels 1 --data_path lung --attacks scale_attk
#python tools/test.py --model_path ./checkpoints/ --mode test --output_path \
#./output/test/ --model DenseNet --batch_size 4 --classes 2 --channels 1 --data_path lung --attacks scale_attk
python tools/test_deeplab.py --output_stride 8 --channels 1 --classes 2 --data_path lung --output_path \
./output/test/ --model deeplabv3plus_resnet101 --adv_model deeplabv3plus_resnet101 \
--attacks scale_attk --batch_size 2 --device 1