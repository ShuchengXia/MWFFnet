close all;
clear all;
clc

%模型重建图像路径
dis = dir("D:\MWFFnet-main\Code\test_result\LOLv1\*.png");
T = cell(1,length(dis));
P = cell(1,length(dis));
  for i=1:length(dis)
    %模型重建图像路径
    TimageName=strcat("D:\MWFFnet-main\Code\test_result\LOLv1\",dis(i).name);
    T{i} = imread(TimageName);
%     T{i}= imresize(T{i}, [400, 600], 'bilinear');
%     T{i} = im2double(T{i});    %将uint8转变为double 否则溢出 数据范围[0,1]
    index = strfind(dis(i).name,'.');
    disP = dis(i).name(1:index-1);
%   正常光照图像
    PimageName=strcat('D:\MWFFnet-main\Code\dataset\LOLV1\eval15\high\',disP,'.png');  %LOLV1正常光照图像

    imgHr = imread(PimageName);
%     imgHr = imgHr(1:size(imgHr,1),5:size(imgHr,2)-4,:);  %图像裁剪
%      imgHr = imgHr(1:size(imgHr,1),1:592,:);  %图像裁剪
    P{i} =  imgHr;
%     P{i} = im2double(P{i});
  end
  
for i = 1:length(dis)
    % 计算当前图像的 PSNR
    psnr_value = bright_psnr(T{i}, P{i});
    psnr(i) = psnr_value;
    
    % 计算当前图像的 SSIM
    ssim_value = ssim(T{i}, P{i});
    Ssim(i) = ssim_value;
    
    % 计算当前图像的颜色差异（CIE Lab）
    color_diff_value = labde(T{i}, P{i});
    ColorDiff(i) = color_diff_value;
    
    % 计算当前图像的 LOE
    loe_value = lo(P{i}, T{i});
    loe(i) = loe_value;
    
end

% 计算并输出所有图像的平均值
PSNR = mean(psnr);
SSIM = mean(Ssim);
ColorDiff = mean(ColorDiff);
Loe = mean(loe);


fprintf('Average PSNR = %.4f\n', PSNR);
fprintf('Average SSIM = %.4f\n', SSIM);
fprintf('Average ColorDiff = %.4f\n', ColorDiff);
fprintf('Average LOE = %.4f\n', Loe);



function [psnr] = bright_psnr(y_true, y_pred)
   y_true=im2double(y_true);  %计算平方时候需要转成double类型，否则uchar类型会丢失数据  
   y_pred=im2double(y_pred);
   if nargin<2      
         D = y_true;  
   else  
        if any(size(y_true)~=size(y_pred))  
        error('The input size is not equal to each other!');  
        end   
        D = y_true - y_pred;   
   end  
    
    mse = mean(mean(mean(D.^2 )));
    max_num = 1.0;
    psnr = 10 * log10(max_num^2 / mse);
end

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

function [color_diff] = labde(y_true, y_pred)
    lab_true = rgb2lab(y_true);
    lab_pred = rgb2lab(y_pred);
    lab_true = im2double(lab_true);  
    lab_pred = im2double(lab_pred);
    
    %Delta-L*
    DL = lab_true(:,:,1) - lab_pred(:,:,1);
    %Delta-a*
    Da = lab_true(:,:,2) - lab_pred(:,:,2);
    %Delta-b
    Db = lab_true(:,:,3) - lab_pred(:,:,3);
    %Delta-E
    color_diff = mean(mean(sqrt( DL.^2 + Da.^2 + Db.^2 )));
end

function [loe] = lo(y_ture,y_pred)
% raw = im2uint8(y_ture);
% enhanceResult = im2uint8(y_pred);
% m=100;n=100;
% rawL = max(raw,[],3);
% [nRow, nCol] = size(rawL);
% r=50/min(nRow,nCol);
% enhanceResultL = max(enhanceResult,[],3);
% N = 100;
% sampleRow = round( linspace(1,nRow,N) ); % 100 samples
% %生成一个第一个元素为a最后一个元素为b的数组n是总采样点数
% %此时产生的数组元素在10^a 到10^b是一个对数曲线。
% sampleCol = round( linspace(1,nCol,N) );
% rawL = rawL(sampleRow, sampleCol); % downsample
% enhanceResultL = enhanceResultL(sampleRow, sampleCol);
% error = 0;
% for i = 1:m 
%     for j = 1:n
%         mapRawOrder = rawL>=rawL(i,j);
%         mapResultOrder = enhanceResultL>=enhanceResultL(i,j);
%         mapError = xor(mapRawOrder,mapResultOrder);
%         error = error + sum(mapError(:));
%     end
% end
% loe = error / ((nRow*r)*(nCol*r));
% % loe = error / (m*n);

raw = im2uint8(y_ture);
enhanceResult = im2uint8(y_pred);
r=raw(:,:,1);
g=raw(:,:,2);
b=raw(:,:,3);
rawL= max( max(r,g),b);
re=enhanceResult(:,:,1);
ge=enhanceResult(:,:,2);
be=enhanceResult(:,:,3);
enhanceResultL= max( max(re,ge),be);
[m, n] = size(rawL);r=50/min(m,n);
M=round(m*r);
N=round(n*r);
sampleRow = round( linspace(1,m,M) ); %  samples
sampleCol = round( linspace(1,n,N) );
rawL = rawL(sampleRow, sampleCol); % downsample
enhanceResultL = enhanceResultL(sampleRow, sampleCol);
error = 0;
for i = 1:M
    for j = 1:N
        mapRawOrder = rawL>=rawL(i,j);
        mapResultOrder =enhanceResultL>=enhanceResultL(i,j);
        mapError = xor(mapRawOrder,mapResultOrder);
        error = error + sum(mapError(:));
    end
end
loe = error /(M*N);
% scoreLOE = num2str(scoreLOE); % 转换为 char 类型
end