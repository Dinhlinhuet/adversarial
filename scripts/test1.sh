#python test.py --model SegNet --channels 3 --model_path ./checkpoints/lung/SegNet.pth --output_path
# ./output/test/ --mode test --attacks DAG_C --adv_model SegNet --data_path lung
#  python test.py --model UNet --channels 3 --model_path ./checkpoints/brain/UNet.pth --output_path \
#  ./output/test/ --mode test --attacks DAG_A --adv_model UNet --data_path brain
#python test_ag.py --best_model ./checkpoints/fundus/AgNet.pth --mode test --data_path fundus --output_path \
#./output/test/ --model AgNet --adv_model UNet --attacks ifgsm --target 1

python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model SegNet --classes 3 --channels 1 --adv_model SegNet --attacks cw --target 1 --mask_type 1 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model SegNet --classes 3 --channels 1 --adv_model UNet --attacks cw --target 1 --mask_type 1 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model SegNet --classes 3 --channels 1 --adv_model DenseNet --attacks cw --target 1 --mask_type 1 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model SegNet --classes 3 --channels 1 --adv_model SegNet --attacks cw --target 1 --mask_type 3 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model SegNet --classes 3 --channels 1 --adv_model UNet --attacks cw --target 1 --mask_type 3 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model SegNet --classes 3 --channels 1 --adv_model DenseNet --attacks cw --target 1 --mask_type 3 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model SegNet --classes 3 --channels 1 --adv_model SegNet --attacks cw --target 2 --mask_type 3 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model SegNet --classes 3 --channels 1 --adv_model UNet --attacks cw --target 2 --mask_type 3 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model SegNet --classes 3 --channels 1 --adv_model DenseNet --attacks cw --target 2 --mask_type 3 \
--suffix c_e1

python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model UNet --classes 3 --channels 1 --adv_model SegNet --attacks cw --target 1 --mask_type 1 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model UNet --classes 3 --channels 1 --adv_model UNet --attacks cw --target 1 --mask_type 1 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model UNet --classes 3 --channels 1 --adv_model DenseNet --attacks cw --target 1 --mask_type 1 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model UNet --classes 3 --channels 1 --adv_model SegNet --attacks cw --target 1 --mask_type 3 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model UNet --classes 3 --channels 1 --adv_model UNet --attacks cw --target 1 --mask_type 3 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model UNet --classes 3 --channels 1 --adv_model DenseNet --attacks cw --target 1 --mask_type 3 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model UNet --classes 3 --channels 1 --adv_model SegNet --attacks cw --target 2 --mask_type 3
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model UNet --classes 3 --channels 1 --adv_model UNet --attacks cw --target 2 --mask_type 3 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model UNet --classes 3 --channels 1 --adv_model DenseNet --attacks cw --target 2 --mask_type 3 \
--suffix c_e1

python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model DenseNet --classes 3 --channels 1 --adv_model SegNet --attacks cw --target 1 --mask_type 1 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model DenseNet --classes 3 --channels 1 --adv_model UNet --attacks cw --target 1 --mask_type 1 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model DenseNet --classes 3 --channels 1 --adv_model DenseNet --attacks cw --target 1 --mask_type 1 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model DenseNet --classes 3 --channels 1 --adv_model SegNet --attacks cw --target 1 --mask_type 3 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model DenseNet --classes 3 --channels 1 --adv_model UNet --attacks cw --target 1 --mask_type 3 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model DenseNet --classes 3 --channels 1 --adv_model DenseNet --attacks cw --target 1 --mask_type 3 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model DenseNet --classes 3 --channels 1 --adv_model SegNet --attacks cw --target 2 --mask_type 3 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model DenseNet --classes 3 --channels 1 --adv_model UNet --attacks cw --target 2 --mask_type 3 \
--suffix c_e1
python test.py --model_path ./checkpoints/ --mode test --data_path octa3mfull --output_path \
./output/test/ --model DenseNet --classes 3 --channels 1 --adv_model DenseNet --attacks cw --target 2 --mask_type 3 \
--suffix c_e1

