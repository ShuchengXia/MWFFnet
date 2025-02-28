function [mssim] = ssim(im1, im2, K, window, L)
%Input : (1) img1: the first image being compared
%        (2) img2: the second image being compared
%        (3) K: constants in the SSIM index formula (see the above
%            reference). defualt value: K = [0.01 0.03]
%        (4) window: local window for statistics (see the above
%            reference). default widnow is Gaussian given by
%            window = fspecial('gaussian', 11, 1.5);
%        (5) L: dynamic range of the images. default: L = 255
%
%Output: (1) mssim: the mean SSIM index value between 2 images.
%            If one of the images being compared is regarded as 
%            perfect quality, then mssim can be considered as the
%            quality measure of the other image.
%            If img1 = img2, then mssim = 1.
%        (2) ssim_map: the SSIM index map of the test image. The map
%            has a smaller size than the input images. The actual size:
%            size(img1) - size(window) + 1.
%
%Default Usage:
%   Given 2 test images img1 and img2, whose dynamic range is 0-255
%
%   [mssim ssim_map] = ssim_index(img1, img2);
%
%Advanced Usage:
%   User defined parameters. For example
%
%   K = [0.05 0.05];
%   window = ones(8);
%   L = 100;
%   [mssim ssim_map] = ssim_index(img1, img2, K, window, L);
%
%See the results:
%
%   mssim                        %Gives the mssim value
%   imshow(max(0, ssim_map).^4)  %Shows the SSIM index map
%
%========================================================================
im1=im2double(im1);
im2=im2double(im2);
im1=im1*255;
im2=im2*255;
% im1 = uint8(im1);
% im2 = uint8(im2);

if size(im1, 3) == 3
    im1 = rgb2ycbcr(im1);
    img1 = im1(:, :, 1);
% R = im1(:,:,1);
% G = im1(:,:,2);
% B = im1(:,:,3);
% 
% img1 = 0.299.*R + 0.587.*G + 0.114.*B;
% yuv(:,:,2) = - 0.1687.*R - 0.3313.*G + 0.5.*B + 128;
% yuv(:,:,3) = 0.5.*R - 0.4187.*G - 0.0813.*B + 128;
end

if size(im2, 3) == 3
    im2 = rgb2ycbcr(im2);
    img2 = im2(:, :, 1);
% R = im2(:,:,1);
% G = im2(:,:,2);
% B = im2(:,:,3);
% 
% img2 = 0.299.*R + 0.587.*G + 0.114.*B;
% yuv(:,:,2) = - 0.1687.*R - 0.3313.*G + 0.5.*B + 128;
% yuv(:,:,3) = 0.5.*R - 0.4187.*G - 0.0813.*B + 128;
end




if (nargin < 2 || nargin > 5)
%    ssim_index = -Inf;
   ssim_map = -Inf;
   return;
end

if (size(img1) ~= size(img2))
%    ssim_index = -Inf;
   ssim_map = -Inf;
   return;
end

[M N] = size(img1);

if (nargin == 2)
   if ((M < 11) || (N < 11))   % 图像大小过小，则没有意义。
%            ssim_index = -Inf;
           ssim_map = -Inf;
      return
   end
   window = fspecial('gaussian', 11, 1.5);        % 参数一个标准偏差1.5，11*11的高斯低通滤波。
   K(1) = 0.01;                                                                      % default settings
   K(2) = 0.03;                                                                      %
   L = 255;                                  %
end

if (nargin == 3)
   if ((M < 11) || (N < 11))
%            ssim_index = -Inf;
           ssim_map = -Inf;
      return
   end
   window = fspecial('gaussian', 11, 1.5);
   L = 255;
   if (length(K) == 2)
      if (K(1) < 0 || K(2) < 0)
%                    ssim_index = -Inf;
                   ssim_map = -Inf;
                   return;
      end
   else
%            ssim_index = -Inf;
           ssim_map = -Inf;
           return;
   end
end

if (nargin == 4)
   [H W] = size(window);
   if ((H*W) < 4 || (H > M) || (W > N))
%            ssim_index = -Inf;
           ssim_map = -Inf;
      return
   end
   L = 255;
   if (length(K) == 2)
      if (K(1) < 0 || K(2) < 0)
%                    ssim_index = -Inf;
                   ssim_map = -Inf;
                   return;
      end
   else
%            ssim_index = -Inf;
           ssim_map = -Inf;
           return;
   end
end

if (nargin == 5)
   [H W] = size(window);
   if ((H*W) < 4 || (H > M) || (W > N))
%            ssim_index = -Inf;
           ssim_map = -Inf;
      return
   end
   if (length(K) == 2)
      if (K(1) < 0 || K(2) < 0)
%                    ssim_index = -Inf;
                   ssim_map = -Inf;
                   return;
      end
   else
%            ssim_index = -Inf;
           ssim_map = -Inf;
           return;
   end
end
%%
C1 = (K(1)*L)^2;    % 计算C1参数，给亮度L（x，y）用。
C2 = (K(2)*L)^2;    % 计算C2参数，给对比度C（x，y）用。
window = window/sum(sum(window)); %滤波器归一化操作。
img1 = double(img1); 
img2 = double(img2);

mu1   = filter2(window, img1, 'valid');  % 对图像进行滤波因子加权
mu2   = filter2(window, img2, 'valid');  % 对图像进行滤波因子加权

mu1_sq = mu1.*mu1;     % 计算出Ux平方值。
mu2_sq = mu2.*mu2;     % 计算出Uy平方值。
mu1_mu2 = mu1.*mu2;    % 计算Ux*Uy值。

sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;  % 计算sigmax （方差）
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;  % 计算sigmay （方差）
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;   % 计算sigmaxy（方差）

if (C1 > 0 && C2 > 0)
   ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
else
   numerator1 = 2*mu1_mu2 + C1;
   numerator2 = 2*sigma12 + C2;
   denominator1 = mu1_sq + mu2_sq + C1;
   denominator2 = sigma1_sq + sigma2_sq + C2;
   ssim_map = ones(size(mu1));
   index = (denominator1.*denominator2 > 0);
   ssim_map(index) = (numerator1(index).*numerator2(index))./(denominator1(index).*denominator2(index));
   index = (denominator1 ~= 0) & (denominator2 == 0);
   ssim_map(index) = numerator1(index)./denominator1(index);
end

mssim = mean(mean(ssim_map));

return
end