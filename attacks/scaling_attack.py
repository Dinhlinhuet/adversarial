import cv2
from torchvision.transforms import Resize
from PIL import Image

from scaleatt.attack.area_attack.area_straight_scale_attack import area_straight_scale_attack


def scl_attack(src_image, tar_image):
    tar_image = cv2.resize(tar_image, (256, 256))  # (256,256) (32,32)
    src_image = cv2.resize(src_image, (1280, 1280))  # (256,256)
    # scale_att: ScaleAttackStrategy = DirectNearestScaleAttack(verbose=True)
    result_attack_image, rewritten_pixels = area_straight_scale_attack(src_img=src_image,
                                                                       tar_image=tar_image, permutation=False,
                                                                       verbose=True)
    # tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)
    result_attack_image = cv2.cvtColor(result_attack_image, cv2.COLOR_BGR2RGB)
    # src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    resize = Resize((256, 256))
    result_attack_image = Image.fromarray(result_attack_image)
    result_attack_image = resize(result_attack_image)
    return result_attack_image
