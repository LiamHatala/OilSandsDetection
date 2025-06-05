"""
custom_transformers.py

Defines custom PyTorch transforms for data augmentation and preprocessing
for segmentation tasks, including:
- Cropping to a region of interest (ROI)
- Adding Gaussian noise
- Combining these with standard torchvision transforms (rotation, shear, resizing)

Usage:
    transform = get_transforms(config)
    transformed_img = transform(image)
"""

import torchvision.transforms.functional as F
from torchvision import models, transforms
import torch

class AddNoise(object):
    """
    Add random Gaussian noise to a tensor image.

    Args:
        noise_std (float): Standard deviation of Gaussian noise.
    """

    def __init__(self, noise_std):
        self.noise_std = noise_std
    
    def __call__(self, tensor):
         """
        Apply Gaussian noise to the input tensor if noise_std > 0.

        Args:
            tensor (Tensor): Image tensor.

        Returns:
            Tensor: Noisy image tensor.
        """

        if  self.noise_std > 0:
            noise = torch.randn_like(tensor) *  self.noise_std
            return tensor + noise
        return tensor

class CropROI_Tensor(object):
    """
    Crop a tensor image to a specific region of interest (ROI).

    Args:
        roi (tuple): A tuple of (left, top, width, height) defining the crop box.
    """

    def __init__(self, roi):  # roi = (left, top, width, height)
        self.left, self.top, self.width, self.height = roi

    def __call__(self, img):
        """
        Crop the input image tensor using the defined ROI.

        Args:
            img (PIL.Image or Tensor): Input image.

        Returns:
            Cropped image.
        """

        return F.crop(img, top=self.top, left=self.left, height=self.height, width=self.width)
def get_transforms(config):
    """
    Create a torchvision transform pipeline based on the given config.

    Args:
        config (dict): Configuration dictionary with keys:
            - "rotation_angle" (float): Max degrees to rotate randomly.
            - "shear_angle" (float): Max shear angle.
            - "image_size" (tuple): Target size (H, W).
            - "noise_std" (float): Std dev for Gaussian noise.
            - "roi" (tuple): Region of interest for cropping (left, top, width, height).

    Returns:
        transform (torchvision.transforms.Compose): Composed transform pipeline.
    """

    aug_rotation = config["rotation_angle"] 
    aug_skew = config["shear_angle"] 
    image_size = config["image_size"]
    aug_noise_std = config["noise_std"]  # Standard deviation for Gaussian noise
    roi = config["roi"]
    transforms_list = [
        transforms.ToTensor(),
        CropROI_Tensor(roi),
        transforms.Resize(size=image_size),
    ]
    
    # Add random rotation
    if aug_rotation > 0:
        transforms_list.append(transforms.RandomRotation(degrees=aug_rotation))
    
    # Add skew (shear)
    if aug_skew > 0:
        transforms_list.append(transforms.RandomAffine(degrees=0, shear=aug_skew))

    transforms_list.append(AddNoise(aug_noise_std))
    
    # Combine all transformations
    transform = transforms.Compose(transforms_list)
    return transform 
