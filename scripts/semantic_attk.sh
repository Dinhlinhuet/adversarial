#python tools/semantic_generating.py --model SegNet --attacks semantic --mask_type 2 --target 0 --data_path fundus \
#--model_path ./checkpoints/ --classes 3 --batch_size 4 --mode test \
#--generator_path ./checkpoints/stargan_/fundus/G.ckpt
#python tools/semantic_generating.py --model SegNet --attacks semantic --mask_type 1 --target 0 --data_path fundus \
#--model_path ./checkpoints/ --classes 3 --batch_size 4 --mode test --device 1 \
#--generator_path ./checkpoints/stargan_/fundus/G.ckpt


#CUDA_LAUNCH_BLOCKING=1
#python tools/semantic_generating.py --model SegNet --attacks semantic --mask_type 5 --target 0 --data_path fundus \
#--model_path ./checkpoints/ --classes 3 --batch_size 4 --mode test --device 1 \
#--generator_path ./checkpoints/stargan_/fundus/G.ckpt

###fundus###
##mode:test##
##type1##
#python tools/semantic_generating.py --model SegNet --attacks semantic --mask_type 3 --target 1 --data_path fundus \
#--model_path ./checkpoints/ --classes 3 --batch_size 4 --mode test \
#--generator_path ./checkpoints/stargan_/fundus/100000-G.ckpt

#python tools/semantic_generating.py --model UNet --attacks semantic --mask_type 5 --target 0 --data_path fundus \
#--model_path ./checkpoints/ --classes 3 --batch_size 4 --mode test \
#--generator_path ./checkpoints/stargan_/fundus/G.ckpt

#python tools/semantic_generating.py --model DenseNet --attacks semantic --mask_type 5 --target 0 --data_path fundus \
#--model_path ./checkpoints/ --classes 3 --batch_size 4 --mode test \
#--generator_path ./checkpoints/stargan_/fundus/G.ckpt

#python tools/semantic_generating.py --model AgNet --attacks semantic --mask_type 5 --target 0 --data_path fundus \
#--model_path ./checkpoints/ --classes 3 --batch_size 4 --mode test \
#--generator_path ./checkpoints/stargan_/fundus/G.ckpt

#python tools/semantic_generating.py --model deeplabv3plus_resnet101 --attacks semantic --mask_type 5 --target 0 --data_path fundus \
#--model_path ./checkpoints/ --classes 3 --batch_size 4 --mode test \
#--generator_path ./checkpoints/stargan_/fundus/G.ckpt
#

##mode:train##
##type1##
python tools/semantic_generating.py --model SegNet --attacks semantic --mask_type 5 --target 0 --data_path fundus \
--model_path ./checkpoints/ --classes 3 --batch_size 4 --mode train \
--generator_path ./checkpoints/stargan_/fundus/G.ckpt

##mode:val##
##type1##
python tools/semantic_generating.py --model SegNet --attacks semantic --mask_type 5 --target 0 --data_path fundus \
--model_path ./checkpoints/ --classes 3 --batch_size 4 --mode val \
--generator_path ./checkpoints/stargan_/fundus/G.ckpt

####brain####
##mode:test##
##type1##
#CUDA_LAUNCH_BLOCKING=1
#python tools/semantic_generating.py --model SegNet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode test --device 1 \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt
#
#python tools/semantic_generating.py --model UNet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode test \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt
#
#python tools/semantic_generating.py --model DenseNet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode test \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt

#python tools/semantic_generating.py --model AgNet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode test \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt

#python tools/semantic_generating.py --model deeplabv3plus_mobilenet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode test \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt

##mode:train##
##type1##
#python tools/semantic_generating.py --model SegNet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode train \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt

#python tools/semantic_generating.py --model UNet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode train \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt
#
#python tools/semantic_generating.py --model DenseNet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode train \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt
#
#python tools/semantic_generating.py --model AgNet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode train \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt
#
#python tools/semantic_generating.py --model deeplabv3plus_mobilenet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode train \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt

##type2##
#python tools/semantic_generating.py --model SegNet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode train \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt
#
#python tools/semantic_generating.py --model UNet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode train \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt
#
#python tools/semantic_generating.py --model DenseNet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode train \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt
#
#python tools/semantic_generating.py --model AgNet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode train \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt
#
#python tools/semantic_generating.py --model deeplabv3plus_mobilenet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode train \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt
#
##mode:val##
##type1##
#python tools/semantic_generating.py --model SegNet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode val \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt
#
#python tools/semantic_generating.py --model UNet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode val \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt
#
#python tools/semantic_generating.py --model DenseNet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode val \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt
#
#python tools/semantic_generating.py --model AgNet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode val \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt
#
#python tools/semantic_generating.py --model deeplabv3plus_mobilenet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode val \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt
#
##type2##
#python tools/semantic_generating.py --model SegNet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode val \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt
#
#python tools/semantic_generating.py --model UNet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode val \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt
#
#python tools/semantic_generating.py --model DenseNet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode val \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt
#
#python tools/semantic_generating.py --model AgNet --attacks semantic --mask_type 5 --target 0 --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode val \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt
#
#python tools/semantic_generating.py --model deeplabv3plus_mobilenet --attacks semantic --mask_type 5 --target 0 \
# --data_path brain \
#--model_path ./checkpoints/ --classes 2 --batch_size 4 --mode val \
#--generator_path ./checkpoints/stargan_/brain/60000-G.ckpt