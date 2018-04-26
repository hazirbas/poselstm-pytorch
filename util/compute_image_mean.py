import numpy as np
from os.path import join as jpath
from PIL import Image

root = './'
dataset = 'StMarysChurch'
imsize = [256, 455] # (H, W)
imlist =  np.loadtxt(jpath(root, dataset, 'dataset_train.txt'),
                    dtype=str, delimiter=' ', skiprows=3, usecols=(0))

mean_image = np.zeros((imsize[0], imsize[1], 3), dtype=np.float)
for i, impath in enumerate(imlist):
    print('%d:%s' % (i+1, impath), end='\r')
    image = Image.open(jpath(root, dataset, impath)).convert('RGB')
    image = image.resize((imsize[1], imsize[0]), Image.BICUBIC)
    mean_image += np.array(image).astype(np.float)

mean_image /= len(imlist)
Image.fromarray(mean_image.astype(np.uint8)).save(jpath(root, dataset, 'mean_image.png'))
np.save(jpath(root, dataset, 'mean_image.npy'), mean_image)
