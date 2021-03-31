import os

def filename(x):
    return int(x[:-4])

img_dir = 'A:/projects/Pytorch_AdversarialAttacks/data/fundus/test/imgs'
ls_names = sorted(os.listdir(img_dir),key=filename)
print('ls', ls_names)