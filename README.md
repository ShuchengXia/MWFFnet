# MWFFnet: Multi-scale Wavelet Feature Fusion Network for Low-light Image Enhancement
The implementation of the paper "Multi-scale Wavelet Feature Fusion Network for Low-light Image Enhancement" published on Computers & Graphics
## Abstract ##
Low-light image enhancement (LLIE) aims to enhance the visibility and quality of low-light images. However, existing methods often struggle to effectively balance global and local image content, resulting in suboptimal results. To address this challenge, we propose a novel multi-scale wavelet feature fusion network (MWFFnet) for low-light image enhancement. Our approach utilizes a U-shaped architecture where traditional downsampling and upsampling operations are replaced by discrete wavelet transform (DWT) and inverse DWT (IDWT), respectively. This strategy helps to reduce the difficulty of learning the complex mapping from low-light images to well-exposed ones. Furthermore, we incorporate a dual transposed attention (DTA) module for each feature scale. DTA effectively captures long-range dependencies between image contents, thus
enhancing the network’s ability to understand intricate image structures. To further improve the enhancement quality, we develop a cross-layer attentional feature fusion (CAFF) module that effectively integrates features from both the encoder and decoder. This mechanism enables the network to leverage contextual information across various levels of representation, resulting in a more comprehensive understanding of the images. Extensive experiments demonstrate that with a reasonable model size, the proposed MWFFnet outperforms several state-of-the-art methods.
## Overview ##
<img src="https://github.com/ShuchengXia/MWFFnet/blob/main/images/overall.png">

## Inference ##
<img src="https://github.com/ShuchengXia/MWFFnet/blob/main/images/test_LOLv1.png" title="LOLv1测试结果">
<img src="https://github.com/ShuchengXia/MWFFnet/blob/main/images/test_LOLv2.png" title="LOLv2测试结果">
