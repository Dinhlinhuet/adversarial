import cv2
import glob

def resize(img_dir,output_dir):
    for im in glob.glob('{}/*.png'.format(img_dir)):
        img = cv2.imread(im)
        resize = cv2.resize(img,(512,512))
        img_name = im.split('/')[-1]
        print('{}/{}'.format(output_dir,img_name))
        cv2.imwrite('{}/{}'.format(output_dir,img_name),resize)

#img_dir='/home/linhld/windowsshare/A_DRIVE/data/medical/lung/dataset.tar/lung_new/masks/'
#out_dir = '/home/linhld/windowsshare/A_DRIVE/data/medical/lung/dataset.tar/resize2/masks/'
img_dir='/home/linhld/windowsshare/A_DRIVE/data/medical/lung/dataset.tar/lung_new/masks/'
out_dir = '/home/linhld/windowsshare/A_DRIVE/data/medical/lung/dataset.tar/resize/masks/'
resize(img_dir,out_dir)