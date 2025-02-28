
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from src.utils import compute_ssim


class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image


class res_Loss (nn.Module):
    def __init__(self, weight=1.0):
        super(res_Loss, self).__init__()
        self.weight = weight
        # self.criterion = nn.MSELoss()           # 默认方式
        self.criterion = nn.L1Loss()               #reduction='mean'
    def forward(self, Y_list, T_list):
        n = len(Y_list)
        loss = 0
        for m in range(0, n):
            # loss += self.weight * self.criterion(Y_list[m], T_list[m])
            loss += self.weight * self.criterion(Y_list[m], T_list[m])
        return loss

class Rec_Loss(nn.Module):
    def __init__(self, weight=1.0):
        super(Rec_Loss, self).__init__()
        self.weight = weight
        self.criterion = nn.L1Loss() #reduction='sum'       # 默认方式
        # self.criterion = nn.MSELoss()
    def forward(self, Y_list, T_list):
        loss = self.weight * self.criterion(Y_list, T_list)
        return loss


def cos_loss(tensor1, tensor2):
    dot_mul = torch.sum(torch.mul(tensor1, tensor2), dim=1)
    tensor1_norm = torch.pow(torch.sum(torch.pow(tensor1, 2), dim=1) + 0.0001, 0.5)
    tensor2_norm = torch.pow(torch.sum(torch.pow(tensor2, 2), dim=1) + 0.0001, 0.5)
    loss = dot_mul / (tensor1_norm * tensor2_norm)
    return 1 - torch.mean(loss)

class Color_Loss(nn.Module):
    def __init__(self,weight=1.0):
        super(Color_Loss, self).__init__()
        self.weight = weight
        self.scale_color = nn.CosineSimilarity(dim=1, eps=1e-6)#reduction='sum'
    def forward(self,Y_list, T_list):
        # loss = self.weight * self.criterion(Y_list[-1],T_list[-1])/Y_list[-1].shape[0]
        # loss = torch.mean(-1 * self.scale_color(Y_list, T_list))/ Y_list.shape[0]
        # loss = torch.mean(-1 * self.scale_color(Y_list, T_list))
        # loss = self.weight * self.criterion(Y_list, T_list) / Y_list.shape[0]
        loss = torch.mean(cos_loss(Y_list, T_list))
        return loss

class Color_Loss_new(nn.Module):
    def __init__(self,weight=1.0):
        super(Color_Loss_new, self).__init__()
        self.weight = weight
    def forward(self, Y_list, T_list):
        b, c, h, w = T_list.shape
        true_reflect_view = T_list.view(b, c, h * w).permute(0, 2, 1)
        pred_reflect_view = Y_list.view(b, c, h * w).permute(0, 2, 1)  # 16 x (512x512) x 3
        true_reflect_norm = torch.nn.functional.normalize(true_reflect_view, dim=-1)
        pred_reflect_norm = torch.nn.functional.normalize(pred_reflect_view, dim=-1)
        cose_value = true_reflect_norm * pred_reflect_norm
        cose_value = torch.sum(cose_value, dim=-1)  # 16 x (512x512)  # print(cose_value.min(), cose_value.max())
        color_loss = torch.mean(1 - cose_value)
        return color_loss

class SSIM_Loss(nn.Module):
    def __init__(self, ssim_weight=1.0):
        super(SSIM_Loss, self).__init__()
        self.weight = ssim_weight
    def forward(self, Y_list, T_list):
        loss = self.weight * compute_ssim(Y_list, T_list)
        return 1.0-loss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss

##############
class CharbonnierLoss2(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss2, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg = torchvision.models.vgg19(pretrained=False)
        pre = torch.load('D:/XinJie_Wei/LearningMaterials/Code/LPTN_demo/LPTN_transformer/vgg19-dcbb9e9d.pth')
        vgg.load_state_dict(pre)
        vgg_pretrained_features = vgg.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, vgg_weight=1.0):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        # self.criterion = nn.L1Loss()
        # self.criterion = nn.L1Loss(reduction='sum')
        self.criterion = nn.L1Loss(reduction='mean')
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.vgg_weight =vgg_weight
    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.vgg_weight * self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
class EdgeLoss(nn.Module):
    def __init__(self, Edge_weight=1.0):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()
        self.Edge_weight = Edge_weight
    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.Edge_weight * self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss
class PSNRLoss(nn.Module):

    def __init__(self, PSNR_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = PSNR_weight
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4
        imdff=pred-target
        rmse=((imdff**2).mean(dim=(1,2,3))+1e-8).sqrt()
        loss=20*torch.log10(1/rmse).mean()
        loss=(50.0-loss)/100.0
        return loss
# def ssim1(input, target, rgb_range=1.0):
#     out_data = model_rgb[0, :].permute(1, 2, 0).cpu().numpy()
#     out_label = target[0, :].permute(1, 2, 0).cpu().numpy()
#     c_s1 = structural_similarity(input[:, :, 0], target[:, :, 0], data_range=rgb_range, sigma=1.5, gaussian_weights=True,
#                                 use_sample_covariance=False)
#     c_s2 = structural_similarity(input[:, :, 1], target[:, :, 1], data_range=rgb_range, sigma=1.5, gaussian_weights=True,
#                                 use_sample_covariance=False)
#     c_s3 = structural_similarity(input[:, :, 2], target[:, :, 2], data_range=rgb_range, sigma=1.5, gaussian_weights=True,
#                                 use_sample_covariance=False)
#
#     return torch.tensor([[(c_s1+c_s2+c_s3)/3]]).float()
class My_loss(nn.Module):
    def __init__(self, Pyr_weight = 0.0,
                 Rec_weight = 1.0,
                 color_weight=0.0,
                 ssim_weight=0.0,
                 vgg_weight=0.0,
                 Edge_weight=0.0,
                 PSNR_weiht=0.0):
        super(My_loss, self).__init__()
        self.Pyr_weight = Pyr_weight
        self.Rec_weight = Rec_weight
        self.color_weight = color_weight
        self.ssim_weight = ssim_weight
        self.vgg_weight = vgg_weight
        self.Edge_weight = Edge_weight
        self.PSNR_weiht = PSNR_weiht

        if self.Pyr_weight != 0:
            self.pyr_loss = res_Loss(Pyr_weight)       # 金字塔损失  默认为MSELOSS
            self.lap_pyramid = Lap_Pyramid_Conv(3)
        if self.Rec_weight != 0:
            self.rec_loss = Rec_Loss(Rec_weight)       # 重建损失    默认为L1LOSS,
        if self.color_weight != 0:
            self.color_loss = Color_Loss(color_weight)           # 颜色损失
        if self.ssim_weight != 0:
            self.ssim_loss = SSIM_Loss(ssim_weight)              # 结构损失
        if self.vgg_weight != 0:
            self.vgg_loss = VGGLoss(vgg_weight)                            # 感知损失
        if self.Edge_weight !=0:
            self.edge_loss =EdgeLoss(Edge_weight)
        if self.PSNR_weiht !=0:
            self.psnr_loss = PSNRLoss(PSNR_weiht)
    # def forward(self, output, pyr_out, GT):
    def forward(self, output, GT):
        myloss = 0
        if self.Pyr_weight != 0:
            pyr_GT = self.lap_pyramid.pyramid_decom(img=GT)
            pyr_out = self.lap_pyramid.pyramid_decom(img=output)
            Y_list = []
            Y_list.append(pyr_out[3])
            Y_list.append(pyr_out[2])
            Y_list.append(pyr_out[1])
            Y_list.append(pyr_out[0])

            T_list = []
            T_list.append(pyr_GT[3])
            T_list.append(pyr_GT[2])
            T_list.append(pyr_GT[1])
            T_list.append(pyr_GT[0])
            pyrloss = self.pyr_loss(Y_list, T_list)
            myloss += pyrloss
        if self.Rec_weight != 0:
            recloss = self.rec_loss(output, GT)
            myloss += recloss
        if self.color_weight != 0:
            colloss = self.color_loss(pyr_out[3], pyr_GT[3])
            myloss += colloss
        if self.ssim_weight != 0:
            ssimloss = self.ssim_loss(output, GT)
            myloss += ssimloss
        if self.vgg_weight != 0:
            vggloss = self.vgg_loss(output, GT)
            myloss += vggloss
        if self.Edge_weight != 0:
            edgeloss = self.edge_loss(output, GT)
            myloss += edgeloss
        if self.PSNR_weiht != 0:
            psnrloss = self.psnr_loss(output, GT)
            myloss += psnrloss
        # myloss = pyrloss + recloss + colloss
        # return recloss,pyrloss,myloss
        #
        # myloss = recloss + vggloss
        # myloss = recloss + colloss
        # myloss = recloss + ssimloss

        return 1, myloss
        # return colloss, myloss
        # return ssimloss, myloss
        # return 1, recloss


class D_loss(nn.Module):
    def __init__(self):
        super(D_loss,self).__init__()
    def forward(self,P_Y,P_T):
        loss = -torch.mean(torch.log(torch.sigmoid(P_T) + 1e-9)) - torch.mean(torch.log(1 - torch.sigmoid(P_Y) + 1e-9))
        return loss



def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2)**2 / float(2 * sigma**2)
    gauss = torch.stack(
        [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss / gauss.sum()

def get_gaussian_kernel(ksize: int, sigma: float) -> torch.Tensor:
    if not isinstance(ksize, int) or ksize % 2 == 0 or ksize <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(ksize))
    window_1d: torch.Tensor = gaussian(ksize, sigma)
    return window_1d

def get_gaussian_kernel2d(ksize, sigma):
    if not isinstance(ksize, tuple) or len(ksize) != 2:
        raise TypeError("ksize must be a tuple of length two. Got {}"
                        .format(ksize))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError("sigma must be a tuple of length two. Got {}"
                        .format(sigma))
    ksize_x, ksize_y = ksize
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y: torch.Tensor = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d

class SSIMLoss(nn.Module):
    def __init__(self,loss_weight=1.0, reduction='mean', window_size: int = 11, max_val: float = 1.0) -> None:
        super(SSIMLoss, self).__init__()
        self.window_size: int = window_size
        self.max_val: float = max_val
        self.reduction: str = reduction
        self.loss_weight = loss_weight

        self.window: torch.Tensor = get_gaussian_kernel2d(
            (window_size, window_size), (1.5, 1.5))
        self.padding: int = self.compute_zero_padding(window_size)

        self.C1: float = (0.01 * self.max_val) ** 2
        self.C2: float = (0.03 * self.max_val) ** 2

    @staticmethod
    def compute_zero_padding(kernel_size: int) -> int:
        """Computes zero padding."""
        return (kernel_size - 1) // 2

    def filter2D(
            self,
            input: torch.Tensor,
            kernel: torch.Tensor,
            channel: int) -> torch.Tensor:
        return F.conv2d(input, kernel, padding=self.padding, groups=channel)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # prepare kernel
        b, c, h, w = img1.shape
        tmp_kernel: torch.Tensor = self.window.to(img1.device).to(img1.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

        # compute local mean per channel
        mu1: torch.Tensor = self.filter2D(img1, kernel, c)
        mu2: torch.Tensor = self.filter2D(img2, kernel, c)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # compute local sigma per channel
        sigma1_sq = self.filter2D(img1 * img1, kernel, c) - mu1_sq
        sigma2_sq = self.filter2D(img2 * img2, kernel, c) - mu2_sq
        sigma12 = self.filter2D(img1 * img2, kernel, c) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
            ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        loss = torch.clamp(1. - ssim_map, min=0, max=1) / 2.

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            pass
        return loss



class ULoss(nn.Module):
    def __init__(self,loss_weight=1.0, reduction='mean',):
        super(ULoss,self).__init__()
        self.edgeloos=EdgeLoss()
        self.ssimloss=SSIMLoss()
        self.psnrloss=PSNRLoss()
        self.loss_weight=loss_weight

    def forward(self, pred, target):
        return 0.33*self.psnrloss(pred,target)+0.33*self.ssimloss(pred,target)+0.1*self.edgeloos(pred,target)

if __name__ =='__main__':
    out = torch.rand([2,3,256,256]).float().to('cuda')
    gt = torch.rand([2, 3, 256, 256]).float().to('cuda')
    # p = torch.rand([2,1]).float()
    # alist = []
    # alist.append(torch.rand([2, 3, 256, 256]).float())
    # alist.append(torch.rand([2, 3, 128, 128]).float())
    # alist.append(torch.rand([2, 3, 64, 64]).float())
    # alist.append(torch.rand([2,3,32,32]).float())
    #
    # total_loss = My_loss()
    # t = total_loss(out,  gt)
    total_loss = ULoss()
    t = total_loss(out, gt)
    # t = total_loss(out,alist,gt)
    print(t)
        