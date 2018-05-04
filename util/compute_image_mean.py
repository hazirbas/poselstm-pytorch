import argparse
import numpy as np
from os.path import join as jpath
from PIL import Image


def params():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, default='datasets/KingsCollege', help='dataset root')
    parser.add_argument('--height', type=int, default=256, help='image height')
    parser.add_argument('--width', type=int, default=455, help='image width')
    parser.add_argument('--save_resized_imgs', action="store_true", default=False, help='save resized train/test images [height, width]')
    return parser.parse_args()

args = params()
dataroot = args.dataroot
imsize = [args.height, args.width] # (H, W)
imlist =  np.loadtxt(jpath(dataroot, 'dataset_train.txt'),
                    dtype=str, delimiter=' ', skiprows=3, usecols=(0))
mean_image = np.zeros((imsize[0], imsize[1], 3), dtype=np.float)
for i, impath in enumerate(imlist):
    print('[%d/%d]:%s' % (i+1, len(imlist), impath), end='\r')
    image = Image.open(jpath(dataroot, impath)).convert('RGB')
    image = image.resize((imsize[1], imsize[0]), Image.BICUBIC)
    mean_image += np.array(image).astype(np.float)

    # save resized training images
    if args.save_resized_imgs:
        image.save(jpath(dataroot, impath))
print()
mean_image /= len(imlist)
Image.fromarray(mean_image.astype(np.uint8)).save(jpath(dataroot, 'mean_image.png'))
np.save(jpath(dataroot, 'mean_image.npy'), mean_image)

# save resized test images
if args.save_resized_imgs:
    imlist =  np.loadtxt(jpath(dataroot, 'dataset_test.txt'),
                        dtype=str, delimiter=' ', skiprows=3, usecols=(0))
    for i, impath in enumerate(imlist):
        print('[%d/%d]:%s' % (i+1, len(imlist), impath), end='\r')
        image = Image.open(jpath(dataroot, impath)).convert('RGB')
        image = image.resize((imsize[1], imsize[0]), Image.BICUBIC)
        image.save(jpath(dataroot, impath))
    print()
