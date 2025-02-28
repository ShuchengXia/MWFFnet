from torch.utils.data import Dataset
import torch
import glob
import numpy as np
import random
import os
import cv2
from pathlib import Path

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']


class LoadImages_LOL(Dataset):
    def __init__(self, low_sRGB_path, img_size=(256, 256), data_type=1, augment=False, normalize=False,is_train=True):
        path = str(Path(low_sRGB_path))
        parent = str(Path(path).parent) + os.sep
        if os.path.isfile(path):  # file
            with open(path, 'r') as f:
                f = f.read().splitlines()
                f = [x.replace('./', parent) if x.startswith('./') else x for x in f]
        elif os.path.isdir(path):  # folder
            f = glob.iglob(path + os.sep + '*.*')
        else:
            raise Exception('%s does not exist' % path)


        self.low_sRGB_files = [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in img_formats]
        if data_type == 1:
            self.normal_sRGB_files = [x.replace('low', 'high') for x in self.low_sRGB_files]
        else:
            self.normal_sRGB_files = [x.replace('Low', 'Normal') for x in self.low_sRGB_files]
            self.normal_sRGB_files = [x.replace('low', 'normal') for x in self.normal_sRGB_files]

        self.crop_shape = img_size
        self.augment = augment
        self.normalize = normalize
        self.is_train = is_train

    def __len__(self):
        return len(self.low_sRGB_files)

    def __getitem__(self, index):
        low_sRGBs = cv2.imread(self.low_sRGB_files[index], flags=-1)
        normal_sRGBs = cv2.imread(self.normal_sRGB_files[index], flags=-1)

        if self.normalize:
            # ratio = ratio[:, None, None, None]
            # low_linearRGBs = low_linearRGBs * ratio
            low_sRGB = cv2.normalize(low_sRGBs.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            normal_sRGB = cv2.normalize(normal_sRGBs.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        else:
            # low_linearRGBs = low_linearRGBs * ratio
            low_sRGB = low_sRGBs / 255.
            normal_sRGB = normal_sRGBs / 255.


        if self.is_train:
            # Image cropping
            low_sRGB, normal_sRGB= crop_image_2(self.crop_shape,low_sRGB, normal_sRGB)
            if self.augment:
                # Random Flip
                low_sRGB, normal_sRGB= random_flip_2(low_sRGB, normal_sRGB)
        else:
            if self.crop_shape:
                low_sRGB, normal_sRGB = crop_image_4(self.crop_shape, low_sRGB, normal_sRGB)
                # low_sRGB, normal_sRGB = crop_image_5(self.crop_shape, low_sRGB, normal_sRGB)

        low_sRGB = low_sRGB[:, :, ::-1].transpose(2, 0, 1)
        low_sRGB = np.ascontiguousarray(low_sRGB)

        normal_sRGB = normal_sRGB[:, :, ::-1].transpose(2, 0, 1)
        normal_sRGB = np.ascontiguousarray(normal_sRGB)

        return {'low_sRGB':torch.from_numpy(low_sRGB), 'normal_sRGB': torch.from_numpy(normal_sRGB),
                'low_sRGB_files': self.low_sRGB_files[index]}


def crop_image_2(crop_shape, img1, img2):
    nw = random.randint(0, img1.shape[0] - crop_shape[0])
    nh = random.randint(0, img1.shape[1] - crop_shape[1])
    crop_img1 = img1[nw:nw + crop_shape[0], nh:nh + crop_shape[1], :]
    crop_img2 = img2[nw:nw + crop_shape[0], nh:nh + crop_shape[1], :]
    return crop_img1, crop_img2

def crop_image_3(crop_shape, img1, img2):
    nw = int((img1.shape[0] % crop_shape) / 2)
    nh = int((img1.shape[1] % crop_shape) / 2)
    crop_img1 = img1[nw:img1.shape[0] - nw, nh:img1.shape[1] - nh, :]
    crop_img2 = img2[nw:img2.shape[0] - nw, nh:img2.shape[1] - nh, :]

    return crop_img1, crop_img2

# Crop with the center
def crop_image_4(crop_shape, img1, img2):
    nw = img1.shape[0] - crop_shape[0]
    nh = img1.shape[1] - crop_shape[1]
    crop_img1 = img1[nw//2:nw//2 + crop_shape[0], nh//2:nh//2 + crop_shape[1], :]
    crop_img2 = img2[nw//2:nw//2 + crop_shape[0], nh//2:nh//2 + crop_shape[1], :]
    return crop_img1, crop_img2

# Pad the input if not_multiple_o
def crop_image_5(crop_shape, img1, img2):
    h, w = img2.shape[1], img2.shape[0]
    mulh, mulw= crop_shape[0], crop_shape[1]

    H, W = ((h + mulh) // mulh) * mulh, ((w + mulw) // mulw) * mulw
    padh = H - h if h % mulh != 0 else 0
    padw = W - w if w % mulh != 0 else 0
    # pad_img1 = F.pad(img1, (0, padw, 0, padh), 'reflect')
    # pad_img2 = F.pad(img2, (0, padw, 0, padh), 'reflect')

    pad_img1 = np.pad(img1, ((0, padw,), (0, padh), (0, 0)), 'reflect')
    pad_img2 = np.pad(img2, ((0, padw,), (0, padh), (0, 0)), 'reflect')
    return pad_img1, pad_img2
def random_flip_2(img1, img2):
    mode = random.randint(0, 7)
    if mode == 0:
        # original
        return img1, img2
    elif mode == 1:
        # flip up and down
        img1 = np.flipud(img1)
        img2 = np.flipud(img2)
    elif mode == 2:
        # rotate counterwise 90 degree
        img1 = np.rot90(img1)
        img2 = np.rot90(img2)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        img1 = np.rot90(img1)
        img2 = np.rot90(img2)

        img1 = np.flipud(img1)
        img2 = np.flipud(img2)
    elif mode == 4:
        # rotate 180 degree
        img1 = np.rot90(img1, k=2)
        img2 = np.rot90(img2, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        img1 = np.rot90(img1, k=2)
        img2 = np.rot90(img2, k=2)

        img1 = np.flipud(img1)
        img2 = np.flipud(img2)
    elif mode == 6:
        # rotate 270 degree
        img1 = np.rot90(img1, k=3)
        img2 = np.rot90(img2, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        img1 = np.rot90(img1, k=3)
        img2 = np.rot90(img2, k=3)

        img1 = np.flipud(img1)
        img2 = np.flipud(img2)
    return img1, img2


def crop_image(crop_shape, img1, img2, img3, img4):
    nw = random.randint(0, img1.shape[0] - crop_shape[0])
    nh = random.randint(0, img1.shape[1] - crop_shape[1])
    crop_img1 = img1[nw:nw + crop_shape[0], nh:nh + crop_shape[1], :]
    crop_img2 = img2[nw:nw + crop_shape[0], nh:nh + crop_shape[1], :]
    crop_img3 = img3[nw:nw + crop_shape[0], nh:nh + crop_shape[1], :]
    crop_img4 = img4[nw:nw + crop_shape[0], nh:nh + crop_shape[1], :]
    return crop_img1, crop_img2, crop_img3, crop_img4


def random_flip(img1, img2, img3, img4):
    mode = random.randint(0, 7)
    if mode == 0:
        # original
        return img1, img2, img3, img4
    elif mode == 1:
        # flip up and down
        img1 = np.flipud(img1)
        img2 = np.flipud(img2)
        img3 = np.flipud(img3)
        img4 = np.flipud(img4)
    elif mode == 2:
        # rotate counterwise 90 degree 逆时针旋转90
        img1 = np.rot90(img1)
        img2 = np.rot90(img2)
        img3 = np.rot90(img3)
        img4 = np.rot90(img4)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        img1 = np.rot90(img1)
        img2 = np.rot90(img2)
        img3 = np.rot90(img3)
        img4 = np.rot90(img4)

        img1 = np.flipud(img1)
        img2 = np.flipud(img2)
        img3 = np.flipud(img3)
        img4 = np.flipud(img4)
    elif mode == 4:
        # rotate 180 degree
        img1 = np.rot90(img1, k=2)
        img2 = np.rot90(img2, k=2)
        img3 = np.rot90(img3, k=2)
        img4 = np.rot90(img4, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        img1 = np.rot90(img1, k=2)
        img2 = np.rot90(img2, k=2)
        img3 = np.rot90(img3, k=2)
        img4 = np.rot90(img4, k=2)

        img1 = np.flipud(img1)
        img2 = np.flipud(img2)
        img3 = np.flipud(img3)
        img4 = np.flipud(img4)
    elif mode == 6:
        # rotate 270 degree
        img1 = np.rot90(img1, k=3)
        img2 = np.rot90(img2, k=3)
        img3 = np.rot90(img3, k=3)
        img4 = np.rot90(img4, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        img1 = np.rot90(img1, k=3)
        img2 = np.rot90(img2, k=3)
        img3 = np.rot90(img3, k=3)
        img4 = np.rot90(img4, k=3)

        img1 = np.flipud(img1)
        img2 = np.flipud(img2)
        img3 = np.flipud(img3)
        img4 = np.flipud(img4)
    return img1, img2, img3, img4

