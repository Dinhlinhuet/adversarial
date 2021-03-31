python test.py --model SegNet --channels 3 --model_path ./checkpoints/lung/SegNet.pth --output_path
 ./output/test/ --mode test --attacks DAG_C --adv_model SegNet --data_path lung
  python test.py --model UNet --channels 3 --mo^Cl_path ./checkpoints/brain/UNet.pth --output_path ./output/test/ --mode test --attacks DAG_A --adv_model UNet --data_path brain