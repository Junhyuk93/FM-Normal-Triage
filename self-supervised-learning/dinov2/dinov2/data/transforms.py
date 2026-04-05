# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Sequence

import torch
from torchvision import transforms
# import albumentations as alb
# from albumentations.pytorch import ToTensorV2
from PIL import ImageEnhance, Image, ImageFilter
import random
class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
# IMAGENET_DEFAULT_MEAN = (0., 0., 0.)
# IMAGENET_DEFAULT_STD = (1, 1, 1)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    transforms_list.extend(
        [
            MaybeToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    return transforms.Compose(transforms_list)
#########################################################
class RandomShiftScaleRotate:
    def __init__(self, shift_limit=0.1, scale_limit=0.2, rotate_limit=5):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit

    def __call__(self, img):
        width, height = img.size
        # Random shift
        max_dx = self.shift_limit * width
        max_dy = self.shift_limit * height
        dx = round(random.uniform(-max_dx, max_dx))
        dy = round(random.uniform(-max_dy, max_dy))
        img = transforms.functional.affine(img, angle=0, translate=(dx, dy), scale=1.0, shear=0)

        # Random scale and rotation
        angle = random.uniform(-self.rotate_limit, self.rotate_limit)
        scale = random.uniform(1 - self.scale_limit, 1 + self.scale_limit)
        return transforms.functional.affine(img, angle=angle, translate=(0, 0), scale=scale, shear=0)

class RandomBlur:
    def __init__(self):
        self.transforms = transforms.RandomChoice([
            transforms.GaussianBlur(kernel_size=3),
            transforms.Lambda(lambda img: img.filter(Image.Filter.BLUR)),  # Replace for other blur types if needed
        ])

    def __call__(self, img):
        return self.transforms(img)

class RandomGamma:
    def __call__(self, img):
        gamma = random.uniform(0.8, 1.2)  # Set appropriate range
        return transforms.functional.adjust_gamma(img, gamma)

class RandomSharpen:
    def __init__(self, lightness=(0.5, 1.5)):
        self.lightness = lightness

    def __call__(self, img):
        # A simple sharpen effect can be applied using PIL ImageEnhance
        enhancer = ImageEnhance.Sharpness(img)
        factor = random.uniform(self.lightness[0], self.lightness[1])
        return enhancer.enhance(factor)

# torchvision equivalent transforms
# def srpark_make_classification_train_transform(
#     *,
#     crop_size: int = 512,
#     interpolation=transforms.InterpolationMode.BICUBIC,
#     hflip_prob: float = 0.5,
#     mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
#     std: Sequence[float] = IMAGENET_DEFAULT_STD,):
#     transform = transforms.Compose([
#     RandomShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5),
#     transforms.RandomChoice([
#         transforms.GaussianBlur(kernel_size=3),
#         transforms.Lambda(lambda img: img.filter(ImageFilter.BLUR)),  # Placeholder for other blur types
#     ]),
#     RandomGamma(),
#     RandomSharpen(lightness=(0.5, 1.5)),
#     transforms.Resize((512, 512), interpolation=interpolation),
#     transforms.ToTensor(),  # Convert to tensor
#     transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))  # Normalize
# ])
#     return transform

def srpark_make_classification_train_transform(
    *,
    crop_size: int = 512,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    # transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
    transforms_list = [transforms.Resize(crop_size, interpolation=interpolation)]
    transforms_list.append(transforms.CenterCrop(crop_size))
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    transforms_list.extend(
        [
            MaybeToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    
    return transforms.Compose(transforms_list)

def srpark_make_classification_train_1K_transform(
    *,
    crop_size: int = 1024,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    # transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
    transforms_list = [transforms.Resize(crop_size, interpolation=interpolation)]
    transforms_list.append(transforms.CenterCrop(crop_size))
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    transforms_list.extend(
        [
            MaybeToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    
    return transforms.Compose(transforms_list)


"""
albumentations 쓰면 
1. 
 File "/opt/conda/envs/dinov2-h100/lib/python3.10/site-packages/albumentations/core/transforms_interface.py", line 190, in update_params_shape
    shape = data["image"].shape if "image" in data else data["images"][0].shape
KeyError: 'images'
2. 
KeyError: 'You have to pass data to augmentations as named arguments, for example: aug(image=image)'
"""
################################################################################

# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)

def srpark_make_classification_eval_transform(
    *,
    resize_size: int = 512,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 512,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)

def srpark_make_classification_eval_1K_transform(
    *,
    resize_size: int = 1024,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 1024,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)

# albumentations
# def srpark_make_classification_eval_transform(
#     *,
#     resize_size: int = 512,
#     interpolation=transforms.InterpolationMode.BICUBIC,
#     crop_size: int = 512,
#     mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
#     std: Sequence[float] = IMAGENET_DEFAULT_STD,
# ) -> transforms.Compose:
#     transforms_list = [
#         # alb.Resize(resize_size, interpolation=interpolation),
#         alb.Resize(512,512),
#                 alb.Normalize(mean=(0,0,0), std=(1,1,1)),
#                 ToTensorV2(),
#     ]
#     return alb.Compose(transforms_list)