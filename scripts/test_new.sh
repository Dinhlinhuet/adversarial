#python test_ag.py --model_path ./checkpoints/ --mode test --output_path \
#./output/test/ --model AgNet --batch_size 4 --classes 3 --data_path fundus --suffix scl_attk
#python test.py --model_path ./checkpoints/ --mode test --output_path \
#./output/test/ --model SegNet --batch_size 4 --classes 3 --data_path fundus --suffix scl_attk
#python test.py --model_path ./checkpoints/ --mode test --output_path \
#./output/test/ --model UNet --batch_size 4 --classes 3 --data_path fundus --suffix scl_attk
#python test.py --model_path ./checkpoints/ --mode test --output_path \
#./output/test/ --model DenseNet --batch_size 4 --classes 3 --data_path fundus --suffix scl_attk

#python test_ag.py --model_path ./checkpoints/ --mode test --data_path fundus --output_path \
#./output/test/ --model AgNet --classes 3 --data_type org --suffix retest

#python test_ag.py --model_path ./checkpoints/ --mode test --output_path \
#./output/test/ --model AgNet --batch_size 4 --classes 2 --data_path brain --suffix scl_attk
#python test.py --model_path ./checkpoints/ --mode test --output_path \
#./output/test/ --model SegNet --batch_size 4 --classes 2 --data_path brain --suffix scl_attk
#python test.py --model_path ./checkpoints/ --mode test --output_path \
#./output/test/ --model UNet --batch_size 4 --classes 2 --data_path brain --suffix scl_attk
#python test.py --model_path ./checkpoints/ --mode test --output_path \
#./output/test/ --model DenseNet --batch_size 4 --classes 2 --data_path brain --suffix scl_attk

python test.py --model_path ./checkpoints/ --mode test --channels 1 --output_path \
./output/test/ --model SegNet --batch_size 4 --classes 3 --data_path octafull --suffix scl_attk
python test.py --model_path ./checkpoints/ --mode test --channels 1  --output_path \
./output/test/ --model UNet --batch_size 4 --classes 3 --data_path octafull --suffix scl_attk
python test.py --model_path ./checkpoints/ --mode test --channels 1  --output_path \
./output/test/ --model DenseNet --batch_size 4 --classes 3 --data_path octafull --suffix scl_attk