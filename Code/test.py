import argparse
import logging
import os
import torch
from collections import OrderedDict
import cv2
import numpy as np
from src import utils
from models.modules.DRSformer import DRSformer
from src.dataset import LoadImages_LOL
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser(
        description='Converting from DLEN.')
    parser.add_argument('--model_dir', '-m',
                        default='./checkpoints/LOLv1.pth',
                        help="Specify the directory of the trained model.",
                        dest='model_dir')
    parser.add_argument('--input_dir', '-i', help='Input image directory',
                        dest='input_dir',
                        default='./dataset/LOLV1/eval15/low/')    # G:/Dataset/eval15/low #G:/Dataset/LOL_v2/Test/Low
    parser.add_argument('--device', '-d', default='cuda',
                        help="Device: cuda or cpu.", dest='device')
    parser.add_argument('--output_dir',
                        default='./test_result/LOLv1/')
    return parser.parse_args()


if __name__ == "__main__":
    import ptflops
    from thop import profile
    args = get_args()
    if args.device.lower() == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    logging.info(f'Using device {device}')

    model = DRSformer().to(device)
    total_params = utils.calc_para(model)
    print(total_params)
    model.eval()
    # data_type=1 denotes test on LOLv1, data_type=2 denotes test on LOLv2
    test = LoadImages_LOL(args.input_dir, img_size=(400, 600), data_type=1, augment=False,
                          normalize=False, is_train=False)
    test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    folder = os.path.exists(args.output_dir)
    if not folder:
        os.makedirs(args.output_dir)
        print("Creat new folder")

    with torch.no_grad():
        checkpoint = torch.load(args.model_dir, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        # model.load_state_dict(checkpoint, strict=False)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        psnr_sRGB = 0
        ssim_sRGB = 0
        img_num = 0
        for batch in test_loader:
            i = 0
            normal_sRGB = batch['normal_sRGB']
            low_sRGB = batch['low_sRGB']
            low_sRGB_files = batch['low_sRGB_files']

            low_sRGB = low_sRGB.to(device=device, dtype=torch.float32)
            low_sRGB = torch.clamp(low_sRGB, 0, 1)
            normal_sRGB = normal_sRGB.to(device=device, dtype=torch.float32)
            normal_sRGB = torch.clamp(normal_sRGB, 0, 1)

            output = model(low_sRGB)
            output = torch.clamp(output, 0, 1)
            # 将填充的大小恢复
            output = output[:, :, :400, :600]
            normal_sRGB= normal_sRGB[:, :, :400, :600]

            # psnr = utils.compute_psnr(output, normal_sRGB)
            # ssim = utils.compute_ssim(output, normal_sRGB)
            # psnr_sRGB = psnr + psnr_sRGB
            # ssim_sRGB = ssim + ssim_sRGB

            output = utils.from_tensor_to_image(output, device=device)
            output = utils.outOfGamutClipping(output)

            in_dir, fn = os.path.split(low_sRGB_files[0])
            name, _ = os.path.splitext(fn)
            outsrgb_name = os.path.join(args.output_dir, name + '.png')
            img_name = name + '.png'
            output = output * 255
            img_num += 1
            cv2.imwrite(outsrgb_name, output.astype(np.uint8))
        # psnr0 = psnr_sRGB / img_num
        # ssim0 = ssim_sRGB / img_num
        # print(psnr0, ssim0)
    print("Test image is done")
    #
    #
