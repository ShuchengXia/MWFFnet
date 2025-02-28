
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import torch.nn.functional as F
import math
import torch.nn as nn
from copy import deepcopy
from math import exp

def outOfGamutClipping(I):
    """ Clips out-of-gamut pixels. """
    I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
    I[I < 0] = 0  # any pixel is below 0, clip it to 0
    return I


def compute_loss(input, target_xyz, rec_xyz, rendered):
    loss = torch.sum(torch.abs(input - rendered) + (
        1.5 * torch.abs(target_xyz - rec_xyz)))/input.size(0)
    return loss

def from_tensor_to_image(tensor, device='cuda'):
    """ converts tensor to image """
    tensor = torch.squeeze(tensor, dim=0)
    if device == 'cpu':
        image = tensor.data.numpy()
    else:
        image = tensor.cpu().data.numpy()
    # CHW to HWC
    image = image.transpose((1, 2, 0))
    image = from_rgb2bgr(image)
    return image

def from_image_to_tensor(image):
    image = from_bgr2rgb(image)
    image = im2double(image)  # convert to double
    image = np.array(image)
    assert len(image.shape) == 3, ('Input image should be 3 channels colored '
                                   'images')
    # HWC to CHW
    image = image.transpose((2, 0, 1))
    # return torch.unsqueeze(torch.from_numpy(image), dim=0)
    return torch.from_numpy(image)


def from_bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB

def from_rgb2bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert from BGR to RGB


def imshow(img, xyz_out=None, srgb_out=None, task=None):
    """ displays images """

    if task.lower() == 'srgb-2-xyz-2-srgb':
        if xyz_out is None:
            raise Exception('XYZ image is not given')
        if srgb_out is None:
            raise Exception('sRGB re-rendered image is not given')

        fig, ax = plt.subplots(1, 3)
        ax[0].set_title('input')
        ax[0].imshow(from_bgr2rgb(img))
        ax[0].axis('off')
        ax[1].set_title('rec. XYZ')
        ax[1].imshow(from_bgr2rgb(xyz_out))
        ax[1].axis('off')
        ax[2].set_title('re-rendered')
        ax[2].imshow(from_bgr2rgb(srgb_out))
        ax[2].axis('off')

    if task.lower() == 'srgb-2-xyz':
        if xyz_out is None:
            raise Exception('XYZ image is not given')

        fig, ax = plt.subplots(1, 2)
        ax[0].set_title('input')
        ax[0].imshow(from_bgr2rgb(img))
        ax[0].axis('off')
        ax[1].set_title('rec. XYZ')
        ax[1].imshow(from_bgr2rgb(xyz_out))
        ax[1].axis('off')

    if task.lower() == 'xyz-2-srgb':
        if srgb_out is None:
            raise Exception('sRGB re-rendered image is not given')

        fig, ax = plt.subplots(1, 2)
        ax[0].set_title('input')
        ax[0].imshow(from_bgr2rgb(img))
        ax[0].axis('off')
        ax[1].set_title('re-rendered')
        ax[1].imshow(from_bgr2rgb(srgb_out))
        ax[1].axis('off')

    if task.lower() == 'pp':
        if srgb_out is None:
            raise Exception('sRGB re-rendered image is not given')

        fig, ax = plt.subplots(1, 2)
        ax[0].set_title('input')
        ax[0].imshow(from_bgr2rgb(img))
        ax[0].axis('off')
        ax[1].set_title('result')
        ax[1].imshow(from_bgr2rgb(srgb_out))
        ax[1].axis('off')

    plt.xticks([]), plt.yticks([])
    plt.show()


def im2double(im):
    """ Returns a double image [0,1] of the uint im. """
    if im[0].dtype == 'uint8':
        max_value = 255
    elif im[0].dtype == 'uint16':
        max_value = 65535
    return im.astype('float') / max_value

def PSNR(img1, img2):
    img1 = img1.detach().cpu()
    img2 = img2.detach().cpu()
    mse = torch.mean(torch.pow(img1 - img2, 2))
    if mse < 1.0e-10:
        return 100
    return 20 * torch.log10(1 / torch.sqrt(mse))

def compute_psnr(y_true, y_pred):
    num = y_true.shape[0]  # batch_size
    psnr = 0.
    for i in range(num):
        mse = torch.mean(
            (torch.abs(y_pred[i, :, :, :] - y_true[i, :, :, :]))**2)
        max_num = 1.0
        if mse < 1.0e-10:
            psnr = 100
        else:
            psnr += 10 * torch.log10(max_num**2 / mse).item()

    return psnr / num
def gaussian(window_size, sigma):
    gauss = torch.tensor([
        exp(-(x - window_size // 2)**2 / float(2 * sigma**2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size,
                               window_size).contiguous()
    return window


def ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(
        img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(
        img1 * img2, window, padding=window_size // 2,
        groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def compute_ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    ssim_value = ssim(img1, img2, window, window_size, channel, size_average)

    return ssim_value.item()

def denorm(img, max_value):
    img = img * float(max_value)
    return img

def norm_img(img, max_value):
    img = img / float(max_value)
    return img


def calc_para(net):
    num_params = 0
    f_params = 0
    m_params = 0
    l_params = 0
    stage = 1
    total_str = 'The number of parameters for each sub-block:\n'

    for param in net.parameters():
        num_params += param.numel()

# 计算网络各部分参数量
    for body in net.named_children():
        res_params = 0
        res_str = []
        for param in body[1].parameters():
            res_params += param.numel()
        res_str = '[{:s}] parameters: {}\n'.format(body[0], res_params)
        total_str = total_str + res_str
        if stage == 1:
            f_params = f_params + res_params
            # if body[0] == 'base_detail':
            #     stage = 2
        elif stage == 2:
            m_params = m_params + res_params
            # if body[0] == 'conv2d':
            #     stage = 3
        elif stage == 3:
            l_params = l_params + res_params
        if 'anchor' in body[0]:     stage += 1

    total_str = total_str + '[total] parameters: {}\n\n'.format(num_params) + \
                '[first_net]\tparameters: {:.4f} M\n'.format(f_params/1e6) + \
                '[middle_net]parameters: {:.4f} M\n'.format(m_params/1e6) + \
                '[last_net]\tparameters: {:.4f} M\n'.format(l_params/1e6) + \
                '[total_net]\tparameters: {:.4f} M\n'.format(num_params/1e6) + \
                '***'
    return total_str
def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
