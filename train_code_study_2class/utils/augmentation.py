from albumentations.core.transforms_interface import (
    ImageOnlyTransform,
    to_tuple,
)
from albumentations.core.transforms_interface import BasicTransform
from albumentations.pytorch.transforms import ToTensorV2

import torch
import copy
import random
import warnings

import albumentations as A
import albumentations.augmentations.functional as F
import numpy as np


class RandomBrightnessContrast(ImageOnlyTransform):
    """Randomly change brightness and contrast of the input image.
    Args:
        brightness_limit ((float, float) or float): factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        contrast_limit ((float, float) or float): factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        brightness_by_max (Boolean): If True adjust contrast by image dtype maximum,
            else adjust contrast by image mean.
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        brightness_limit=0.2,
        contrast_limit=0.2,
        brightness_by_max=True,
        always_apply=False,
        p=0.5,
    ):
        super(RandomBrightnessContrast, self).__init__(always_apply, p)
        self.brightness_limit = to_tuple(brightness_limit)
        self.contrast_limit = to_tuple(contrast_limit)
        self.brightness_by_max = brightness_by_max

    def apply(self, img, alpha=1.0, beta=0.0, **params):
        if img.shape[-1] == 4:
            channel_image_to_skip = np.expand_dims(copy.deepcopy(img[:,:,0]), -1)

            img = F.brightness_contrast_adjust(img[:,:,1:], alpha, beta, self.brightness_by_max)
            img = np.concatenate((channel_image_to_skip, img), axis=-1)
            del channel_image_to_skip
            return img
        elif img.shape[-1] == 5:
            channel_image_to_skip = copy.deepcopy(img[:,:,0:2])

            img = F.brightness_contrast_adjust(img[:,:,2:], alpha, beta, self.brightness_by_max)
            img = np.concatenate((channel_image_to_skip, img), axis=-1)
            del channel_image_to_skip
            return img
        else:
            return F.brightness_contrast_adjust(img, alpha, beta, self.brightness_by_max)

    def get_params(self):
        return {
            "alpha": 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
            "beta": 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1]),
        }

    def get_transform_init_args_names(self):
        return ("brightness_limit", "contrast_limit", "brightness_by_max")

class RandomBrightness(RandomBrightnessContrast):
    """Randomly change brightness of the input image.
    Args:
        limit ((float, float) or float): factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self, limit=0.2, always_apply=False, p=0.5):
        super(RandomBrightness, self).__init__(
            brightness_limit=limit, contrast_limit=0, always_apply=always_apply, p=p
        )
        warnings.warn(
            "This class has been deprecated. Please use RandomBrightnessContrast",
            FutureWarning,
        )

    def get_transform_init_args(self):
        return {"limit": self.brightness_limit}


def get_study_transform(img_size):
    train_transform = A.Compose([
        A.Resize(img_size,img_size),
        A.HorizontalFlip(p=0.5),
        RandomBrightness(limit=0.1, p=0.75),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, border_mode=0, p=0.75),
        A.Cutout(max_h_size=int(img_size * 0.3), max_w_size=int(img_size * 0.3), num_holes=1, p=0.75),
        ToTensorV2(p=1.0),
    ])
    val_transform = A.Compose([
        A.Resize(img_size,img_size),
        ToTensorV2(p=1.0),
    ])
    return train_transform, val_transform
