import argparse

def get_args():

    parser = argparse.ArgumentParser('argument')
    parser.add_argument('--data_path', dest='data_path' ,type=str,
                      default='./data/', help='data path')
    parser.add_argument('--attack_path', dest='attack_path' ,type=str,
                      default=None, help='the path of adversarial attack examples')
    parser.add_argument('--model_path', dest='model_path' ,type=str, default='./checkpoints/',
                      help='model_path')
    parser.add_argument('--classes', dest='classes', default=2, type=int,
                      help='number of classes')
    parser.add_argument('--channels', dest='channels', default=3, type=int,
                      help='number of channels')
    parser.add_argument('--width', dest='width', default=256, type=int,
                      help='image width')
    parser.add_argument('--height', dest='height', default=256, type=int,
                      help='image height')
    parser.add_argument('--GroupNorm', action="store_true", default= True,
                      help='decide to use the GroupNorm')
    parser.add_argument('--BatchNorm', action="store_false", default = False,
                      help='decide to use the BatchNorm')
    parser.add_argument('--model', dest='model', type=str,
                      help='model name(UNet, SegNet, DenseNet)')
    parser.add_argument('-m', '--middle', dest='middle',  default=False, action="store_true",
                      help='whether to have middle connection')
    parser.add_argument('--attacks', dest='attacks', type=str, default="",
                      help='attack types: dag')
    parser.add_argument('--target', dest='target', default='0', type=str,
                      help='target class')
    parser.add_argument('--mask_type', dest='mask_type', default="", type=str,
                      help='adv mask')
    parser.add_argument('--adv_model', dest='adv_model', type=str,default='',
                      help='model name(UNet, SegNet, DenseNet)')
    parser.add_argument('--mode', dest='mode', type=str, default='',
                                            help='mode test origin or adversarial')
    parser.add_argument('--data_type', dest='data_type', type=str,default='',
                      help='org or DAG')
    parser.add_argument('--source_version', dest='source_version', default='0', type=str,
                        help='source_version')
    parser.add_argument('--gpus', dest='gpus',type=int,
                          default=1, help='gpu or cpu')
    parser.add_argument('--batch_size', dest='batch_size', default=2, type=int,
                      help='batch size')
    parser.add_argument('--suffix', dest='suffix', type=str,
                      default='', help='suffix to purpose')
    parser.add_argument('--output_path', dest='output_path', type=str,
                      default='./output', help='output_path')
    parser.add_argument('--adv_path', dest='adv_path', type=str,
                      default='./output/adv/', help='adv_path')
    parser.add_argument('--save-dir', default='./checkpoints/denoiser/', type=str, metavar='SAVE',
                        help='directory to save checkpoint (default: none)')
    parser.add_argument('--target_model', default='./checkpoints/', type=str, metavar='PATH',
                      help='path to target model (default: none)')
    parser.add_argument('--epochs', dest='epochs', default=50, type=int,
                      help='number of epochs')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                                              help='manual epoch number (useful on restarts)')
    parser.add_argument('--generator_path', dest='generator_path', type=str,
                      default='checkpoints/stargan_/', help='generator_path')
    parser.add_argument('--denoiser_path', dest='denoiser_path', type=str,
                      default='checkpoints/denoiser/', help='denoiser_path')
    parser.add_argument('--denoise_output', dest='denoise_output', type=str,
                      default='./output/denoised_imgs/', help='denoise_output')
    parser.add_argument('--device', dest='device', default=0, type=int,
                      help='device index number')
    parser.add_argument('--transfer', dest='transfer', default=False, action="store_true",
                      help='transferability')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument('--exp_folder', type=str, default='./output/adv/exps', help='Experiment folder to store all output.')
    args = parser.parse_args()
    print('model with middle connection', args.middle)
    return args