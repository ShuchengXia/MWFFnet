import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import os
from skimage.util import img_as_ubyte
from natsort import natsorted
from glob import glob
import argparse
import cv2
from models.modules.DRSformer_arch import DRSformer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Demo Low-light Image Enhancement')
parser.add_argument('--input_dir', default='low/', type=str, help='Input images')
parser.add_argument('--result_dir', default='./test_result/', type=str, help='Directory for results')
parser.add_argument('--weights', default='model_best.pth', type=str, help='Path to weights')
parser.add_argument('--scale_factor', default=0.5, type=float, help='Scale factor for downscaling (e.g., 0.5 for half size)')

args = parser.parse_args()


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


inp_dir = args.input_dir
out_dir = args.result_dir

os.makedirs(out_dir, exist_ok=True)

files = natsorted(glob(os.path.join(inp_dir, '*.bmp')) +
                  glob(os.path.join(inp_dir, '*.JPG')) +
                  glob(os.path.join(inp_dir, '*.png')) +
                  glob(os.path.join(inp_dir, '*.PNG')) +
                  glob(os.path.join(inp_dir, '*.jpg')) +
                  glob(os.path.join(inp_dir, '*.jpeg')))

if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")

# Load the model and weights
model = DRSformer().to("cuda")
checkpoint = torch.load(args.weights, map_location="cuda")
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()

print('restoring images......')

mul = 16  # Padding multiple
scale_factor = args.scale_factor  # Downscaling factor
index = 0

for file_ in files:
    img = Image.open(file_).convert('RGB')
    input_ = TF.to_tensor(img).unsqueeze(0).cuda()

    # Get original dimensions
    h, w = input_.shape[2], input_.shape[3]


    # Directly enhance the whole image if no resizing is needed
    H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
    padh = H - h if h % mul != 0 else 0
    padw = W - w if w % mul != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
    with torch.no_grad():
        restored = model(input_)
    restored = restored[:, :, :h, :w]

    # Convert to image format and save
    restored = torch.clamp(restored, 0, 1)
    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored = img_as_ubyte(restored[0])

    f = os.path.splitext(os.path.split(file_)[-1])[0]
    save_img(os.path.join(out_dir, f + '.png'), restored)
    index += 1
    print('%d/%d' % (index, len(files)))

print(f"Files saved at {out_dir}")
print('finish !')
