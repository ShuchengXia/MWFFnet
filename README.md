# MWFFnet: Multi-scale Wavelet Feature Fusion Network for Low-light Image Enhancement
The implementation of the paper "Multi-scale Wavelet Feature Fusion Network for Low-light Image Enhancement" published on Computers & Graphics
## Abstract ##
Low-light image enhancement (LLIE) aims to enhance the visibility and quality of low-light images. However, existing methods often struggle to effectively balance global and local image content, resulting in suboptimal results. To address this challenge, we propose a novel multi-scale wavelet feature fusion network (MWFFnet) for low-light image enhancement. Our approach utilizes a U-shaped architecture where traditional downsampling and upsampling operations are replaced by discrete wavelet transform (DWT) and inverse DWT (IDWT), respectively. This strategy helps to reduce the difficulty of learning the complex mapping from low-light images to well-exposed ones. Furthermore, we incorporate a dual transposed attention (DTA) module for each feature scale. DTA effectively captures long-range dependencies between image contents, thus
enhancing the network’s ability to understand intricate image structures. To further improve the enhancement quality, we develop a cross-layer attentional feature fusion (CAFF) module that effectively integrates features from both the encoder and decoder. This mechanism enables the network to leverage contextual information across various levels of representation, resulting in a more comprehensive understanding of the images. Extensive experiments demonstrate that with a reasonable model size, the proposed MWFFnet outperforms several state-of-the-art methods.
## Overview ##
<img src="https://github.com/ShuchengXia/MWFFnet/blob/main/images/overall.png">

## Test ##
Please see the Readme file in Code/Test 

## Inference ##
### LOLv1&LOLv2测试结果 ###
<img src="https://github.com/ShuchengXia/MWFFnet/blob/main/images/test_LOLv1.png" width="427" height="442"><img src="https://github.com/ShuchengXia/MWFFnet/blob/main/images/test_LOLv2.png" width="427" height="442">

## Citation ##
If you find our repo useful for your research, please consider citing this paper and our previous work
```
ARTICLE{PRNet,
     author={Ling, Mingyang and Chang, Kan and Huang, Mengyuan and Li, Hengxin and Dang, Shuping and Li Baoxin},
     journal={IEEE Transactions on Computational Imaging},
     title={PRNet: Pyramid Restoration Network for RAW Image Super-Resolution},
     year={2024},
     volume={10},
     pages={479-495},
     doi={10.1109/TCI.2024.3374084}
}
```
