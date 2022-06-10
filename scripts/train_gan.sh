#python tools/train_stargan.py --mode train --image_size 256 \
#               --c_dim 1 --g_conv_dim 128 --d_conv_dim 128 --batch_size 16 --device 1\
#               --sample_dir output/debug/stargan/samples/fundus --log_dir ./checkpoints/stargan/logs/fundus \
#               --model_save_dir ./checkpoints/stargan_/fundus \
#               --data_path fundus \
##               --resume_iters 150000 \
##               --num_iters 200000

#python tools/train_stargan.py --mode train --image_size 256 \
#               --c_dim 1 --g_conv_dim 64 --d_conv_dim 64 --batch_size 8 --device 0\
#               --sample_dir output/debug/stargan/samples/fundus --log_dir ./checkpoints/stargan/logs/fundus \
#               --model_save_dir ./checkpoints/stargan_/fundus \
#               --data_path fundus \
#               --resume_iters 150000 \
#               --num_iters 200000

#python tools/train_stargan.py --mode train --image_size 256 \
#               --c_dim 1 --g_conv_dim 128 --batch_size 16 --device 1\
#               --sample_dir output/debug/stargan/samples/brain --log_dir ./checkpoints/stargan/logs/brain \
#               --model_save_dir ./checkpoints/stargan_/brain \
#               --data_path brain \
##               --resume_iters 60000

#python tools/train_stargan.py --mode train --image_size 256 \
#               --c_dim 1 --g_conv_dim 128 --d_conv_dim 128 --batch_size 8 --device 0 --channels 1\
#               --sample_dir output/debug/stargan/samples/lung --log_dir ./checkpoints/stargan/logs/lung \
#               --model_save_dir ./checkpoints/stargan_/lung \
#               --data_path lung

python tools/train_stargan.py --mode train --image_size 256 \
               --c_dim 1 --g_conv_dim 128 --d_conv_dim 128 --batch_size 4 --device 0 --channels 1\
               --sample_dir output/debug/stargan/samples/octafull --log_dir ./checkpoints/stargan/logs/octafull \
               --model_save_dir ./checkpoints/stargan_/octafull \
               --data_path octafull