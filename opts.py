from optparse import OptionParser

def get_args():

    parser = OptionParser()
    parser.add_option('--data_path', dest='data_path' ,type='string',
                      default='./data/', help='data path')
    parser.add_option('--attack_path', dest='attack_path' ,type='string',
                      default=None, help='the path of adversarial attack examples')
    parser.add_option('--model_path', dest='model_path' ,type='string', default='./checkpoints/',
                      help='model_path')
    parser.add_option('--classes', dest='classes', default=2, type='int',
                      help='number of classes')
    parser.add_option('--channels', dest='channels', default=3, type='int',
                      help='number of channels')
    parser.add_option('--width', dest='width', default=256, type='int',
                      help='image width')
    parser.add_option('--height', dest='height', default=256, type='int',
                      help='image height')
    parser.add_option('--GroupNorm', action="store_true", default= True,
                      help='decide to use the GroupNorm')
    parser.add_option('--BatchNorm', action="store_false", default = False,
                      help='decide to use the BatchNorm')
    parser.add_option('--model', dest='model', type='string',
                      help='model name(UNet, SegNet, DenseNet)')
    parser.add_option('--attacks', dest='attacks', type='string', default="",
                      help='attack types: Rician, DAG_A, DAG_B, DAG_C')
    parser.add_option('--target', dest='target', default='0', type='string',
                      help='target class')
    parser.add_option('--mask_type', dest='mask_type', default="", type='string',
                      help='adv mask')
    parser.add_option('--adv_model', dest='adv_model', type='string',default='',
                      help='model name(UNet, SegNet, DenseNet)')
    parser.add_option('--mode', dest='mode', type='string', default='',
                                            help='mode test origin or adversarial')
    parser.add_option('--data_type', dest='data_type', type='string',default='',
                      help='org or DAG')
    parser.add_option('--gpus', dest='gpus',type='int',
                          default=1, help='gpu or cpu')
    parser.add_option('--batch_size', dest='batch_size', default=2, type='int',
                      help='batch size')
    parser.add_option('--suffix', dest='suffix', type='string',
                      default='', help='suffix to purpose')
    parser.add_option('--output_path', dest='output_path', type='string',
                      default='./output', help='output_path')
    parser.add_option('--save-dir', default='./checkpoints/denoiser/', type='string', metavar='SAVE',
                        help='directory to save checkpoint (default: none)')
    parser.add_option('--target_model', default='./checkpoints/', type='string', metavar='PATH',
                      help='path to target model (default: none)')
    parser.add_option('--epochs', dest='epochs', default=50, type='int',
                      help='number of epochs')
    parser.add_option('--resume', default='', type='string', metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_option('--start-epoch', default=0, type=int, metavar='N',
                                              help='manual epoch number (useful on restarts)')
    parser.add_option('--denoiser_path', dest='denoiser_path', type='string',
                      default='checkpoints/denoiser/', help='denoiser_path')
    parser.add_option('--denoise_output', dest='denoise_output', type='string',
                      default='./output/denoised_imgs/', help='denoise_output')
    parser.add_option('--device', dest='device', default=0, type='int',
                      help='device index number')
    parser.add_option('--device1', dest='device1', default=-1, type='int',
                      help='device1 index number')
    parser.add_option('--device2', dest='device2', default=-1, type='int',
                      help='device2 index number')
    parser.add_option('--device3', dest='device3', default=-1, type='int',
                      help='device3 index number')
    parser.add_option('--device4', dest='device4', default=-1, type='int',
                      help='device4 index number')

    (options, args) = parser.parse_args()
    return options