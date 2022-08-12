import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import kornia
import torch.nn.functional as F
from typing import Tuple, List, Union
import cv2

eps_l2_norm = 1e-10
keynet_config = {

    'KeyNet_default_config':
        {
            # Key.Net Model
            'num_filters': 8,
            'num_levels': 3,
            'kernel_size': 5,

            # Trained weights
            'weights_detector': r'C:\Users\DELL\Desktop\projs\BaseMethod_PY\weights\keynet\keynet_pytorch.pth',
            'weights_descriptor': r'C:\Users\DELL\Desktop\projs\BaseMethod_PY\weights\keynet\HyNet_LIB.pth',

            # Extraction Parameters
            'nms_size': 15,
            'pyramid_levels': 4,
            'up_levels': 1,
            'scale_factor_levels': np.sqrt(2),
            's_mult': 22,
        },
}

class feature_extractor(nn.Module):
    '''
        It loads both, the handcrafted and learnable blocks
    '''
    def __init__(self):
        super(feature_extractor, self).__init__()

        self.hc_block = handcrafted_block()
        self.lb_block = learnable_block()

    def forward(self, x):
        x_hc = self.hc_block(x)
        x_lb = self.lb_block(x_hc)
        return x_lb

class handcrafted_block(nn.Module):
    '''
        It defines the handcrafted filters within the Key.Net handcrafted block
    '''
    def __init__(self):
        super(handcrafted_block, self).__init__()

    def forward(self, x):

        sobel = kornia.spatial_gradient(x)
        dx, dy = sobel[:, :, 0, :, :], sobel[:, :, 1, :, :]

        sobel_dx = kornia.spatial_gradient(dx)
        dxx, dxy = sobel_dx[:, :, 0, :, :], sobel_dx[:, :, 1, :, :]

        sobel_dy = kornia.spatial_gradient(dy)
        dyy = sobel_dy[:, :, 1, :, :]

        hc_feats = torch.cat([dx, dy, dx**2., dy**2., dx*dy, dxy, dxy**2., dxx, dyy, dxx*dyy], dim=1)

        return hc_feats

def conv_blck(in_channels=8, out_channels=8, kernel_size=5,
              stride=1, padding=2, dilation=1):
    '''
    Default learnable convolutional block.
    '''
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))

class learnable_block(nn.Module):
    '''
        It defines the learnable blocks within the Key.Net
    '''
    def __init__(self, in_channels=10):
        super(learnable_block, self).__init__()

        self.conv0 = conv_blck(in_channels)
        self.conv1 = conv_blck()
        self.conv2 = conv_blck()

    def forward(self, x):
        x = self.conv2(self.conv1(self.conv0(x)))
        return x

# Utility from Kornia: https://kornia.readthedocs.io/en/latest/_modules/kornia/geometry/transform/pyramid.html
def _get_pyramid_gaussian_kernel() -> torch.Tensor:
    """Utility function that return a pre-computed gaussian kernel."""
    return torch.tensor([[
        [1., 4., 6., 4., 1.],
        [4., 16., 24., 16., 4.],
        [6., 24., 36., 24., 6.],
        [4., 16., 24., 16., 4.],
        [1., 4., 6., 4., 1.]
    ]]) / 256.

def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalizes both derivative and smoothing kernel.
    """
    if len(input.size()) < 2:
        raise TypeError("input should be at least 2D tensor. Got {}"
                        .format(input.size()))
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))

def compute_padding(kernel_size: Tuple[int, int]) -> List[int]:
    """Computes padding tuple."""
    # 4 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    assert len(kernel_size) == 2, kernel_size
    computed = [(k - 1) // 2 for k in kernel_size]
    return [computed[1], computed[1], computed[0], computed[0]]

def filter2D(input: torch.Tensor, kernel: torch.Tensor,
             border_type: str = 'reflect',
             normalized: bool = False) -> torch.Tensor:
    r"""Function that convolves a tensor with a kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    indepentdently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output reaims in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(B, kH, kW)`.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): If True, kernel will be L1 normalized.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Input kernel type is not a torch.Tensor. Got {}"
                        .format(type(kernel)))

    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    if not len(kernel.shape) == 3:
        raise ValueError("Invalid kernel shape, we expect BxHxW. Got: {}"
                         .format(kernel.shape))

    borders_list: List[str] = ['constant', 'reflect', 'replicate', 'circular']
    if border_type not in borders_list:
        raise ValueError("Invalid border_type, we expect the following: {0}."
                         "Got: {1}".format(borders_list, border_type))

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.to(input.device).to(input.dtype)
    tmp_kernel = tmp_kernel.repeat(c, 1, 1, 1)
    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)
    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]
    padding_shape: List[int] = compute_padding((height, width))
    input_pad: torch.Tensor = F.pad(input, padding_shape, mode=border_type)

    # convolve the tensor with the kernel
    return F.conv2d(input_pad, tmp_kernel, padding=0, stride=1, groups=c)


def custom_pyrdown(input: torch.Tensor, factor: float = 2., border_type: str = 'reflect', align_corners: bool = False) -> torch.Tensor:
    r"""Blurs a tensor and downsamples it.

    Args:
        input (tensor): the tensor to be downsampled.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        align_corners(bool): interpolation flag. Default: False. See
        https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate for detail.

    Return:
        torch.Tensor: the downsampled tensor.

    Examples:
        >>> input = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        >>> pyrdown(input, align_corners=True)
        tensor([[[[ 3.7500,  5.2500],
                  [ 9.7500, 11.2500]]]])
    """
    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")
    kernel: torch.Tensor = _get_pyramid_gaussian_kernel()
    b, c, height, width = input.shape
    # blur image
    x_blur: torch.Tensor = filter2D(input, kernel, border_type)

    # downsample.
    out: torch.Tensor = F.interpolate(x_blur, size=(int(height // factor), int(width // factor)), mode='bilinear',
                                      align_corners=align_corners)
    return out


class KeyNet(nn.Module):
    '''
    Key.Net model definition
    '''
    def __init__(self, keynet_conf):
        super(KeyNet, self).__init__()

        num_filters = keynet_conf['num_filters']
        self.num_levels = keynet_conf['num_levels']
        kernel_size = keynet_conf['kernel_size']
        padding = kernel_size // 2

        self.feature_extractor = feature_extractor()
        self.last_conv = nn.Sequential(nn.Conv2d(in_channels=num_filters*self.num_levels,
                                                 out_channels=1, kernel_size=kernel_size, padding=padding),
                                       nn.ReLU(inplace=True))

    def forward(self, x):
        """
        x - input image
        """
        shape_im = x.shape
        for i in range(self.num_levels):
            if i == 0:
                feats = self.feature_extractor(x)
            else:
                x = custom_pyrdown(x, factor=1.2)
                feats_i = self.feature_extractor(x)
                feats_i = F.interpolate(feats_i, size=(shape_im[2], shape_im[3]), mode='bilinear', align_corners=True)
                feats = torch.cat([feats, feats_i], dim=1)

        scores = self.last_conv(feats)
        return scores


class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-6, is_bias=True, is_scale=True, is_eps_leanable=False):
        """
        weight = gamma, bias = beta

        beta, gamma:
            Variables of shape [1, 1, 1, C]. if TensorFlow
            Variables of shape [1, C, 1, 1]. if PyTorch
        eps: A scalar constant or learnable variable.
        """
        super(FRN, self).__init__()

        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_leanable = is_eps_leanable
        self.is_bias = is_bias
        self.is_scale = is_scale


        self.weight = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.bias = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        if is_eps_leanable:
            self.eps = nn.parameter.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.is_eps_leanable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={init_eps}'.format(**self.__dict__)

    def forward(self, x):
        """
        0, 1, 2, 3 -> (B, H, W, C) in TensorFlow
        0, 1, 2, 3 -> (B, C, H, W) in PyTorch
        TensorFlow code
            nu2 = tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True)
            x = x * tf.rsqrt(nu2 + tf.abs(eps))

            # This Code include TLU function max(y, tau)
            return tf.maximum(gamma * x + beta, tau)
        """
        # Compute the mean norm of activations per channel.
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)

        # Perform FRN.
        x = x * torch.rsqrt(nu2 + self.eps.abs())

        # Scale and Bias
        if self.is_scale:
            x = self.weight * x
        if self.is_bias:
            x = x + self.bias
        return x


class TLU(nn.Module):
    def __init__(self, num_features):
        """max(y, tau) = max(y - tau, 0) + tau = ReLU(y - tau) + tau"""
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.zeros_(self.tau)
        nn.init.constant_(self.tau, -1)

    def extra_repr(self):
        return 'num_features={num_features}'.format(**self.__dict__)

    def forward(self, x):
        return torch.max(x, self.tau)

class HyNet(nn.Module):
    """
    HyNet model definition.
    The FRN and TLU layer are from the papaer
    `Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks`
    https://github.com/yukkyo/PyTorch-FilterResponseNormalizationLayer
    """
    def __init__(self, is_bias=True, is_bias_FRN=True, dim_desc=128, drop_rate=0.3):
        super(HyNet, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate

        self.layer1 = nn.Sequential(
            FRN(1, is_bias=is_bias_FRN),
            TLU(1),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False)
        )

        self.desc_norm = nn.Sequential(
            nn.LocalResponseNorm(2 * self.dim_desc, alpha=2 * self.dim_desc, beta=0.5, k=0)
        )

        return

    def forward(self, x):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7]:
            x = layer(x)

        x = self.desc_norm(x + eps_l2_norm)
        x = x.view(x.size(0), -1)
        return x


def initialize_networks(conf):
    '''
    It loads the detector and descriptor models
    :param conf: It contains the configuration and weights path of the models
    :return: Key.Net and HyNet models
    '''
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    detector_path = conf['weights_detector']
    descriptor_path = conf['weights_descriptor']

    # Define keynet_model model
    keynet_model = KeyNet(conf)
    checkpoint = torch.load(detector_path)
    keynet_model.load_state_dict(checkpoint['state_dict'])
    keynet_model = keynet_model.to(device)
    keynet_model.eval()

    desc_model = HyNet()
    checkpoint = torch.load(descriptor_path)
    desc_model.load_state_dict(checkpoint)

    desc_model = desc_model.to(device)
    desc_model.eval()

    return keynet_model, desc_model

class NonMaxSuppression(torch.nn.Module):
    '''
        NonMaxSuppression class
    '''
    def __init__(self, thr=0.0, nms_size=5):
        nn.Module.__init__(self)
        padding = nms_size // 2
        self.max_filter = torch.nn.MaxPool2d(kernel_size=nms_size, stride=1, padding=padding)
        self.thr = thr

    def forward(self, scores):

        # local maxima
        maxima = (scores == self.max_filter(scores))

        # remove low peaks
        maxima *= (scores > self.thr)

        return maxima.nonzero().t()[2:4]


def remove_borders(score_map, borders):
    '''
    It removes the borders of the image to avoid detections on the corners
    '''
    shape = score_map.shape
    mask = torch.ones_like(score_map)

    mask[:, :, 0:borders, :] = 0
    mask[:, :, :, 0:borders] = 0
    mask[:, :, shape[2] - borders:shape[2], :] = 0
    mask[:, :, :, shape[3] - borders:shape[3]] = 0

    return mask*score_map


def raise_error_if_laf_is_not_valid(laf: torch.Tensor) -> None:
    """Auxilary function, which verifies that input is a torch.tensor of [BxNx2x3] shape

    Args:
        laf
    """
    laf_message: str = "Invalid laf shape, we expect BxNx2x3. Got: {}".format(laf.shape)
    if not torch.is_tensor(laf):
        raise TypeError("Laf type is not a torch.Tensor. Got {}"
                        .format(type(laf)))
    if len(laf.shape) != 4:
        raise ValueError(laf_message)
    if laf.size(2) != 2 or laf.size(3) != 3:
        raise ValueError(laf_message)
    return


def scale_laf(laf: torch.Tensor, scale_coef: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Multiplies region part of LAF ([:, :, :2, :2]) by a scale_coefficient.
    So the center, shape and orientation of the local feature stays the same, but the region area changes.

    Args:
        laf: (torch.Tensor): tensor [BxNx2x3] or [BxNx2x2].
        scale_coef: (torch.Tensor): broadcastable tensor or float.


    Returns:
        torch.Tensor: tensor  BxNx2x3 .

    Shape:
        - Input: :math: `(B, N, 2, 3)`
        - Input: :math: `(B, N,)` or ()
        - Output: :math: `(B, N, 1, 1)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> scale = 0.5
        >>> output = kornia.scale_laf(input, scale)  # BxNx2x3
    """
    if (type(scale_coef) is not float) and (type(scale_coef) is not torch.Tensor):
        raise TypeError(
            "scale_coef should be float or torch.Tensor "
            "Got {}".format(type(scale_coef)))
    raise_error_if_laf_is_not_valid(laf)
    centerless_laf: torch.Tensor = laf[:, :, :2, :2]
    return torch.cat([scale_coef * centerless_laf, laf[:, :, :, 2:]], dim=3)


def laf_from_center_scale_ori(xy: torch.Tensor, scale: torch.Tensor, ori: torch.Tensor) -> torch.Tensor:
    """Returns orientation of the LAFs, in radians. Useful to create kornia LAFs from OpenCV keypoints

    Args:
        xy: (torch.Tensor): tensor [BxNx2].
        scale: (torch.Tensor): tensor [BxNx1x1].
        ori: (torch.Tensor): tensor [BxNx1].

    Returns:
        torch.Tensor: tensor  BxNx2x3 .
    """
    names = ['xy', 'scale', 'ori']
    for var_name, var, req_shape in zip(names,
                                        [xy, scale, ori],
                                        [("B", "N", 2), ("B", "N", 1, 1), ("B", "N", 1)]):
        if not isinstance(var, torch.Tensor):
            raise TypeError("{} type is not a torch.Tensor. Got {}"
                            .format(var_name, type(var)))
        if len(var.shape) != len(req_shape):  # type: ignore  # because it does not like len(tensor.shape)
            raise TypeError(
                "{} shape should be must be [{}]. "
                "Got {}".format(var_name, str(req_shape), var.size()))
        for i, dim in enumerate(req_shape):  # type: ignore # because it wants typing for dim
            if dim is not int:
                continue
            if var.size(i) != dim:
                raise TypeError(
                    "{} shape should be must be [{}]. "
                    "Got {}".format(var_name, str(req_shape), var.size()))
    unscaled_laf: torch.Tensor = torch.cat([kornia.angle_to_rotation_matrix(ori.squeeze(-1)),
                                            xy.unsqueeze(-1)], dim=-1)
    laf: torch.Tensor = scale_laf(unscaled_laf, scale)
    return laf


def normalize_laf(LAF: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """Normalizes LAFs to [0,1] scale from pixel scale. See below:
        >>> B,N,H,W = images.size()
        >>> MIN_SIZE = min(H,W)
        [a11 a21 x]
        [a21 a22 y]
        becomes:
        [a11/MIN_SIZE a21/MIN_SIZE x/W]
        [a21/MIN_SIZE a22/MIN_SIZE y/H]

    Args:
        LAF: (torch.Tensor).
        images: (torch.Tensor) images, LAFs are detected in

    Returns:
        LAF: (torch.Tensor).

    Shape:
        - Input: :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, 2, 3)`
    """
    raise_error_if_laf_is_not_valid(LAF)
    n, ch, h, w = images.size()
    w = float(w)
    h = float(h)
    min_size = min(h, w)
    coef = torch.ones(1, 1, 2, 3).to(LAF.dtype) / min_size
    coef[0, 0, 0, 2] = 1.0 / w
    coef[0, 0, 1, 2] = 1.0 / h
    coef.to(LAF.device)
    return coef.expand_as(LAF) * LAF


def get_laf_scale(LAF: torch.Tensor) -> torch.Tensor:
    """Returns a scale of the LAFs

    Args:
        LAF: (torch.Tensor): tensor [BxNx2x3] or [BxNx2x2].

    Returns:
        torch.Tensor: tensor  BxNx1x1 .

    Shape:
        - Input: :math: `(B, N, 2, 3)`
        - Output: :math: `(B, N, 1, 1)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> output = kornia.get_laf_scale(input)  # BxNx1x1
    """
    raise_error_if_laf_is_not_valid(LAF)
    eps = 1e-10
    out = LAF[..., 0:1, 0:1] * LAF[..., 1:2, 1:2] - LAF[..., 1:2, 0:1] * LAF[..., 0:1, 1:2] + eps
    return out.abs().sqrt()


def denormalize_laf(LAF: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """De-normalizes LAFs from scale to image scale.
        >>> B,N,H,W = images.size()
        >>> MIN_SIZE = min(H,W)
        [a11 a21 x]
        [a21 a22 y]
        becomes
        [a11*MIN_SIZE a21*MIN_SIZE x*W]
        [a21*MIN_SIZE a22*MIN_SIZE y*H]

    Args:
        LAF: (torch.Tensor).
        images: (torch.Tensor) images, LAFs are detected in

    Returns:
        LAF: (torch.Tensor).

    Shape:
        - Input: :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, 2, 3)`
    """
    raise_error_if_laf_is_not_valid(LAF)
    n, ch, h, w = images.size()
    w = float(w)
    h = float(h)
    min_size = min(h, w)
    coef = torch.ones(1, 1, 2, 3).to(LAF.dtype) * min_size
    coef[0, 0, 0, 2] = w
    coef[0, 0, 1, 2] = h
    coef.to(LAF.device)
    return coef.expand_as(LAF) * LAF


def generate_patch_grid_from_normalized_LAF(img: torch.Tensor,
                                            LAF: torch.Tensor,
                                            PS: int = 32) -> torch.Tensor:
    """Helper function for affine grid generation.

    Args:
        img: (torch.Tensor) images, LAFs are detected in
        LAF: (torch.Tensor).
        PS (int) -- patch size to be extracted

    Returns:
        grid: (torch.Tensor).

    Shape:
        - Input: :math:`(B, CH, H, W)`,  :math:`(B, N, 2, 3)`
        - Output:  :math:`(B, N, PS, PS)`
    """
    raise_error_if_laf_is_not_valid(LAF)
    B, N, _, _ = LAF.size()
    num, ch, h, w = img.size()

    # norm, then renorm is needed for allowing detection on one resolution
    # and extraction at arbitrary other
    LAF_renorm = denormalize_laf(LAF, img)

    grid = F.affine_grid(LAF_renorm.view(B * N, 2, 3),
                         [B * N, ch, PS, PS], align_corners=True)
    grid[..., :, 0] = 2.0 * grid[..., :, 0].clone() / float(w) - 1.0
    grid[..., :, 1] = 2.0 * grid[..., :, 1].clone() / float(h) - 1.0
    return grid


def extract_patches_from_pyramid(img: torch.Tensor,
                                 laf: torch.Tensor,
                                 PS: int = 32,
                                 normalize_lafs_before_extraction: bool = True) -> torch.Tensor:
    """Extract patches defined by LAFs from image tensor.
    Patches are extracted from appropriate pyramid level

    Args:
        laf: (torch.Tensor).
        images: (torch.Tensor) images, LAFs are detected in
        PS: (int) patch size, default = 32
        normalize_lafs_before_extraction (bool):  if True, lafs are normalized to image size, default = True

    Returns:
        patches: (torch.Tensor)  :math:`(B, N, CH, PS,PS)`
    """
    raise_error_if_laf_is_not_valid(laf)
    if normalize_lafs_before_extraction:
        nlaf: torch.Tensor = normalize_laf(laf, img)
    else:
        nlaf = laf
    B, N, _, _ = laf.size()
    num, ch, h, w = img.size()
    scale = 2.0 * get_laf_scale(denormalize_laf(nlaf, img)) / float(PS)
    half: float = 0.5
    pyr_idx = (scale.log2() + half).relu().long()
    cur_img = img
    cur_pyr_level = 0
    out = torch.zeros(B, N, ch, PS, PS).to(nlaf.dtype).to(nlaf.device)
    while min(cur_img.size(2), cur_img.size(3)) >= PS:
        num, ch, h, w = cur_img.size()
        # for loop temporarily, to be refactored
        for i in range(B):
            scale_mask = (pyr_idx[i] == cur_pyr_level).squeeze()
            if (scale_mask.float().sum()) == 0:
                continue
            scale_mask = (scale_mask > 0).view(-1)
            grid = generate_patch_grid_from_normalized_LAF(
                cur_img[i:i + 1],
                nlaf[i:i + 1, scale_mask, :, :],
                PS)
            patches = F.grid_sample(cur_img[i:i + 1].expand(grid.size(0), ch, h, w), grid,  # type: ignore
                                    padding_mode="border", align_corners=True)
                                    #padding_mode="border")
            out[i].masked_scatter_(scale_mask.view(-1, 1, 1, 1), patches)
        cur_img = kornia.pyrdown(cur_img)
        cur_pyr_level += 1
    return out


def extract_ms_feats(keynet_model, desc_model, image, factor, s_mult, device,
                     num_kpts_i=1000, nms=None, down_level=0, up_level=False, im_size=[]):
    '''
    Extracts the features for a specific scale level from the pyramid
    :param keynet_model: Key.Net model
    :param desc_model: HyNet model
    :param image: image as a PyTorch tensor
    :param factor: rescaling pyramid factor
    :param s_mult: Descriptor area multiplier
    :param device: GPU or CPU
    :param num_kpts_i: number of desired keypoints in the level
    :param nms: nums size
    :param down_level: Indicates if images needs to go down one pyramid level
    :param up_level: Indicates if image is an upper scale level
    :param im_size: Original image size
    :return: It returns the local features for a specific image level
    '''

    if down_level and not up_level:
        image = custom_pyrdown(image, factor=factor)
        _, _, nh, nw = image.shape
        factor = (im_size[0]/nw, im_size[1]/nh)
    elif not up_level:
        factor = (1., 1.)

    # src kpts:
    with torch.no_grad():
        det_map = keynet_model(image)
    det_map = remove_borders(det_map, borders=15)

    kps = nms(det_map)
    c = det_map[0, 0, kps[0], kps[1]]
    sc, indices = torch.sort(c, descending=True)
    indices = indices[torch.where(sc > 0.)]
    kps = kps[:, indices[:num_kpts_i]]
    kps_np = torch.cat([kps[1].view(-1, 1).float(), kps[0].view(-1, 1).float(), c[indices[:num_kpts_i]].view(-1, 1).float()],
        dim=1).detach().cpu().numpy()
    num_kpts = len(kps_np)
    kp = torch.cat([kps[1].view(-1, 1).float(), kps[0].view(-1, 1).float()],dim=1).unsqueeze(0).cpu()
    s = s_mult * torch.ones((1, num_kpts, 1, 1))
    src_laf = laf_from_center_scale_ori(kp, s, torch.zeros((1, num_kpts, 1)))

    # HyNet takes images on the range [0, 255]
    patches = extract_patches_from_pyramid(255*image.cpu(), src_laf, PS=32, normalize_lafs_before_extraction=True)[0]

    if len(patches) > 1000:
        for i_patches in range(len(patches)//1000+1):
            if i_patches == 0:
                descs = desc_model(patches[:1000].to(device))
            else:
                descs_tmp = desc_model(patches[1000*i_patches:1000*(i_patches+1)].to(device))
                descs = torch.cat([descs, descs_tmp], dim=0)
        descs = descs.cpu().detach().numpy()
    else:
        descs = desc_model(patches.to(device)).cpu().detach().numpy()

    kps_np[:, 0] *= factor[0]
    kps_np[:, 1] *= factor[1]

    return kps_np, descs, image.to(device)


def compute_kpts_desc(im_path, keynet_model, desc_model, conf, device, num_points):
    '''
    The script computes Multi-scale kpts and desc of an image.

    :param im_path: path to image
    :param keynet_model: Detector model
    :param desc_model: Descriptor model
    :param conf: Configuration file to load extraction settings
    :param device: GPU or CPU
    :param num_points: Number of total local features
    :return: Keypoints and descriptors associated with the image
    '''

    # Load extraction configuration
    pyramid_levels = conf['pyramid_levels']
    up_levels = conf['up_levels']
    scale_factor_levels = conf['scale_factor_levels']
    s_mult = conf['s_mult']
    nms_size = conf['nms_size']
    nms = NonMaxSuppression(nms_size=nms_size)

    # Compute points per level
    point_level = []
    tmp = 0.0
    factor_points = (scale_factor_levels ** 2)
    levels = pyramid_levels + up_levels + 1
    for idx_level in range(levels):
        tmp += factor_points ** (-1 * (idx_level - up_levels))
        point_level.append(num_points * factor_points ** (-1 * (idx_level - up_levels)))

    point_level = np.asarray(list(map(lambda x: int(x / tmp), point_level)))

    im_np = np.asarray(cv2.imread(im_path, 0) / 255., np.float32)
    print(im_np.shape)

    im = torch.from_numpy(im_np).unsqueeze(0).unsqueeze(0)
    im = im.to(device)

    if up_levels:
        im_up = torch.from_numpy(im_np).unsqueeze(0).unsqueeze(0)
        im_up = im_up.to(device)

    src_kp = []
    _, _, h, w = im.shape
    # Extract features from the upper levels
    for idx_level in range(up_levels):

        num_points_level = point_level[len(point_level) - pyramid_levels - 1 - (idx_level+1)]

        # Resize input image
        up_factor = scale_factor_levels ** (1 + idx_level)
        nh, nw = int(h * up_factor), int(w * up_factor)
        up_factor_kpts = (w/nw, h/nh)
        im_up = F.interpolate(im_up, (nh, nw), mode='bilinear', align_corners=False)

        src_kp_i, src_dsc_i, im_up = extract_ms_feats(keynet_model, desc_model, im_up, up_factor_kpts,
                                                      s_mult=s_mult, device=device, num_kpts_i=num_points_level,
                                                      nms=nms, down_level=idx_level+1, up_level=True, im_size=[w, h])

        src_kp_i = np.asarray(list(map(lambda x: [x[0], x[1], (1 / scale_factor_levels) ** (1 + idx_level), x[2]], src_kp_i)))

        if src_kp == []:
            src_kp = src_kp_i
            src_dsc = src_dsc_i
        else:
            src_kp = np.concatenate([src_kp, src_kp_i], axis=0)
            src_dsc = np.concatenate([src_dsc, src_dsc_i], axis=0)

    # Extract features from the downsampling pyramid
    for idx_level in range(pyramid_levels + 1):

        num_points_level = point_level[idx_level]
        if idx_level > 0 or up_levels:
            res_points = int(np.asarray([point_level[a] for a in range(0, idx_level + 1 + up_levels)]).sum() - len(src_kp))
            num_points_level = res_points

        src_kp_i, src_dsc_i, im = extract_ms_feats(keynet_model, desc_model, im, scale_factor_levels, s_mult=s_mult,
                                                   device=device, num_kpts_i=num_points_level, nms=nms,
                                                   down_level=idx_level, im_size=[w, h])

        src_kp_i = np.asarray(list(map(lambda x: [x[0], x[1], scale_factor_levels ** idx_level, x[2]], src_kp_i)))

        if len(src_kp) == 0:
            src_kp = src_kp_i
            src_dsc = src_dsc_i
        else:
            src_kp = np.concatenate([src_kp, src_kp_i], axis=0)
            src_dsc = np.concatenate([src_dsc, src_dsc_i], axis=0)

    return src_kp, src_dsc


if __name__ == "__main__":

    num_kpts = 5000
    img_path = r"E:\datasets\bop_datasets\test\9\img\000049_000001.png"

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Read Key.Net model and extraction configuration
    conf = keynet_config['KeyNet_default_config']
    keynet_model, desc_model = initialize_networks(conf)

    xys, desc = compute_kpts_desc(img_path, keynet_model, desc_model, conf, device, num_points=5000)
    img = cv2.imread(img_path)
    kps = []
    for x, y, scale, score in xys:
      print(x, y, scale, score)
      kp = cv2.KeyPoint(x, y, 0)
      kps.append(kp)
    img_keypoints = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    cv2.drawKeypoints(img, kps, img_keypoints)
    #cv2.imwrite("keynet.png", img_keypoints)
    cv2.imshow("keynet", img_keypoints)
    cv2.waitKey(0)
    print()
