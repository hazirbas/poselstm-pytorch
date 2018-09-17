import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy
import torch

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Resize(opt.loadSize, Image.BICUBIC))
        # transform_list.append(transforms.Lambda(
        #     lambda img: __scale_width(img, opt.loadSize)))
        if opt.isTrain:
            transform_list.append(transforms.RandomCrop(opt.fineSize))
        else:
            transform_list.append(transforms.CenterCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_posenet_transform(opt, mean_image):
    transform_list = []
    transform_list.append(transforms.Resize(opt.loadSize, Image.BICUBIC))
    transform_list.append(transforms.Lambda(
        lambda img: __subtract_mean(img, mean_image)))
    transform_list.append(transforms.Lambda(
        lambda img: __crop_image(img, opt.fineSize, opt.isTrain)))
    transform_list.append(transforms.Lambda(
        lambda img: __to_tensor(img)))
    return transforms.Compose(transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)

def __subtract_mean(img, mean_image):
    if mean_image is None:
        return numpy.array(img).astype('float')
    return numpy.array(img).astype('float') - mean_image.astype('float')

def __crop_image(img, size, isTrain):
    h, w = img.shape[0:2]
    # w, h = img.size
    if isTrain:
        if w == size and h == size:
            return img
        x = numpy.random.randint(0, w - size)
        y = numpy.random.randint(0, h - size)
    else:
        x = int(round((w - size) / 2.))
        y = int(round((h - size) / 2.))
    return img[y:y+size, x:x+size, :]
    # return img.crop((x, y, x + size, y + size))

def __to_tensor(img):
    return torch.from_numpy(img.transpose((2, 0, 1)))
