"""
oil_sands_dataset.py

Custom dataset for loading image and mask pairs for semantic segmentation,
tailored for oil sands level detection. Assumes directory structure with group-level folders
and mask-label mappings.

Includes dataset splitting, transform filtering for masks, and optional noise/cropping.
"""

import os
import numpy as np
from torchvision import transforms as v2
from transforms.custom_transforms import AddNoise, CropROI_Tensor
from typing import Optional, Callable, Literal
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import torch



class Dataset(VisionDataset):
    """
    Custom PyTorch Dataset for loading images and corresponding segmentation masks.

    Args:
        root (str): Root path containing image and mask subfolders.
        img_folder (str): Subdirectory name under root for images.
        mask_folder (str): Subdirectory name under root for masks.
        set_type (Literal["train", "test"]): Dataset split to use.
        color_mode (Literal["RGB", "L"], optional): Image color mode. Defaults to "RGB".
        transforms (Callable, optional): Transformations applied to images and masks.
        seed (int, optional): Random seed for reproducible splitting. Defaults to 100.
        val_split (float, optional): Train/validation split ratio. Defaults to 0.8.
    """
    def __init__(
        self,
        root: str,
        img_folder: str,
        mask_folder: str,
        set_type: Literal["train", "test"],
        color_mode: Literal["RGB", "L"] = "RGB",
        transforms: Optional[Callable] = None,
        seed: int = 100,
        val_split: float = 0.8,
    ):
        super().__init__(root, transforms)

        # Separate transform pipline for masks
        self.mask_transforms = self.__filter_trfms(transforms)

        img_folder_path = os.path.join(root, img_folder)
        mask_folder_path = os.path.join(root, mask_folder)

        assert os.path.exists(
            img_folder_path
        ), "image folder path doesn't exist, check where it is defined and from where the script is called"
        assert os.path.exists(
            mask_folder_path
        ), "mask folder path doesn't exist, check where it is defined and from where the script is called"

        # Get the list of all group directories
        group_dirs = sorted(os.listdir(img_folder_path))
        label_dirs = sorted(os.listdir(mask_folder_path))

        # Only keep groups that have corresponding labels
        valid_groups = []
        for group in group_dirs:
            label_group = f"labels_group_{group.split('_')[1]}"
            if label_group in label_dirs:
                valid_groups.append(group)
                break

        self.img_pth_names = []
        self.mask_pth_names = []

        for group in valid_groups:
            image_path = os.path.join(img_folder_path, group)
            mask_path = os.path.join(
                mask_folder_path, "labels_group_" + group.split("_")[1]
            )

            for filename in os.listdir(mask_path):
                _, ext = os.path.splitext(filename)
                if ext.lower() == ".jpg":
                    filename_image = filename.split(".")[0] + ".jpg"
                    full_path_image = os.path.join(image_path, filename_image)
                    self.img_pth_names.append(full_path_image)

                    full_path_mask = os.path.join(mask_path, filename)
                    self.mask_pth_names.append(full_path_mask)
        self.img_pth_names = np.array(self.img_pth_names)
        self.mask_pth_names = np.array(self.mask_pth_names)


        self.color_mode = color_mode
        # Shuffle and split the dataset
        if seed:
            np.random.seed(seed)
            indices = np.arange(len(self.img_pth_names))
            np.random.shuffle(indices)

            self.img_pth_names = self.img_pth_names[indices]
            self.mask_pth_names = self.mask_pth_names[indices]

        # split dataset and keep only train or test part based on `set_type`
        num_images = int(val_split * len(self.img_pth_names))

        if set_type == "train":
            self.img_pth_names = self.img_pth_names[:num_images]
            self.mask_pth_names = self.mask_pth_names[:num_images]

        else:
            self.img_pth_names = self.img_pth_names[num_images:]
            self.mask_pth_names = self.mask_pth_names[num_images:]

    def __len__(self):
        return len(self.img_pth_names)

    def __getitem__(self, index: int):
        """
        Load an image-mask pair and apply transformations.

        Returns:
            dict: Dictionary with keys "image" and "mask", both as tensors.
        """

        img_pth = self.img_pth_names[index]
        mask_pth = self.mask_pth_names[index]
        img = Image.open(img_pth).convert(self.color_mode)
        mask = Image.open(mask_pth).convert("L")
        sample = {"image": img, "mask": mask}
        if self.transforms:
            sample["image"] = self.transforms(sample["image"])
            sample["mask"] = self.mask_transforms(sample["mask"])
            
            # Ensuring mask is binary - 0,1
            sample["mask"] = torch.where(sample["mask"] > 0, 1, 0)

        return sample
    
    def __filter_trfms(self, trfm: Callable):
        """
        Remove unsuitable transforms for masks, like color jitter or noise.

        Args:
            trfm (Callable): Composed transforms.

        Returns:
            Compose: Filtered transforms suitable for masks.
        """

        trfm_list = trfm.transforms
        invalid_trfms = [v2.ColorJitter, AddNoise, v2.Normalize]

        for trfm in trfm_list:
            if any([isinstance(trfm, i) for i in invalid_trfms]):
                trfm_list.remove(trfm)
        return v2.Compose(trfm_list)
