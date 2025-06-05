"""
train.py

This script is used to train a DeepLabv3 segmentation model on oil sands hopper level images.
It uses PyTorch with custom metrics and transformation pipelines. The model is trained to 
predict pixel-wise masks and infer oil level categories (Low, Adequate, High) from those masks.

"""


from typing import Optional, Callable, Literal
import os
import time
import copy
import tqdm
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torcheval.metrics.functional import binary_f1_score, binary_auroc
from metrics.custom_metrics import level_accuracy
import numpy as np
import torch
from PIL import Image

from data.oil_sands_dataset import Dataset
from tqdm import tqdm

from transforms.custom_transforms import AddNoise, CropROI_Tensor, get_transforms
import optuna
import torch
from torch.optim import Adam
from torchvision import transforms as v2
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
from torch.serialization import add_safe_globals
from torch._dynamo.eval_frame import OptimizedModule
import torchvision.transforms.functional as F

from models.model import createDeepLabv3

def train(model, dataloaders, config):
    """
    Trains the given DeepLabV3 model on the provided dataloaders using MSE loss.

    Args:
        model (torch.nn.Module): The DeepLabV3 segmentation model.
        dataloaders (list): List of DataLoader objects for training and testing sets.
        config (dict): Training configuration dictionary with keys:
            - num_epochs (int)
            - batch_size (int)
            - learning rate, noise_std, image_size, roi, etc.

    Returns:
        None. Saves the best model and logs metrics to CSV.
    """
    torch.backends.cudnn.enabled = True
    num_epochs = config["num_epochs"]
    level_dict = {0: "Low Level", 1: "Adequate Level", 2: "High Level"}
    metrics = {"f1_score": binary_f1_score, "auroc": binary_auroc, "level_accuracy": level_accuracy}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max= num_epochs)

    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()
    since = time.time()
    best_loss = 1e10
    # Use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Initialize the log file for training and testing loss and metrics
    fieldnames = (
        ["epoch", "train_loss", "test_loss"]
        + [f"train_{m}" for m in metrics.keys()]
        + [f"test_{m}" for m in metrics.keys()]
    )
    with open(os.path.join(bpath, "log.csv"), "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print("-" * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        for phase in [0,1]: # 0 = train, 1 = test
            if phase == 0:
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            progress_bar = tqdm(dataloaders[phase], desc=f"Epoch {epoch} - Phase {phase_to_str(phase)}", dynamic_ncols=True)
            total_loss = 0
            for sample in progress_bar:
                inputs = sample["image"].to(device)
                masks = torch.squeeze(sample["mask"].to(device))
                labels = torch.zeros(masks.shape, dtype=torch.long, device=device)
                labels = torch.where(masks > 0, 1, 0)
                labels = labels.permute((0, 1, 2)).to(torch.float32)
                level_labels = getBatchLabels(labels)

                # Calculate the accuracy of the labels 

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 0):
                    outputs = model(inputs)
                    out = outputs["out"].permute((1, 0, 2, 3))[0, ...]
                    level_outputs = getBatchLabels(out)
                    loss = criterion(out, labels)
                    total_loss += loss
                    y_pred_level = level_outputs
                    y_true_level = level_labels

                    y_pred = out
                    y_true = labels

                    for name, metric in metrics.items():
                        if name == "f1_score":
                            # Use a classification threshold of 0.1
                            batchsummary[f"{phase_to_str(phase)}_{name}"].append(
                                metric(y_true.flatten(), y_pred.flatten())
                            )
                        elif name == "level_accuracy":
                            level_acc = metric(y_pred_level, y_true_level).item()
                            batchsummary[f"{phase_to_str(phase)}_{name}"].append(level_acc)
                        else:
                            batchsummary[f"{phase_to_str(phase)}_{name}"].append(
                                metric(y_pred.flatten(), masks.flatten())
                            )

                    # backward + optimize only if in training phase
                    if phase == 0:
                        loss.backward()
                        optimizer.step()
                progress_bar.set_postfix(loss=loss.item())
            batchsummary["epoch"] = epoch
            avg_loss = total_loss/len(dataloaders[phase].dataset)
            relative_loss = (avg_loss**0.5)
            epoch_loss = relative_loss*100
            batchsummary[f"{phase_to_str(phase)}_loss"] = epoch_loss.item()
            print("{} Loss: {:.4f}".format(phase_to_str(phase), total_loss))
            total_loss = 0
        if phase == 0:
            scheduler.step()
        for field in fieldnames[3:]:
            batchsummary[field] = torch.Tensor(batchsummary[field]).mean().cpu()
        print(batchsummary)
        with open(os.path.join(bpath, "log.csv"), "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    torch.save(model, "best_mdl_wts.pt")


def get_Pretrained_model(model_path):
    """
    Loads a pretrained model from a given path.

    Args:
        model_path (str): Path to the saved model (.pt file).

    Returns:
        torch.nn.Module: Loaded PyTorch model on CUDA (if available).
    """
    model = torch.load(model_path, map_location=torch.device('cuda'), weights_only=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Move model to the right device (GPU or CPU)
    model = model.to(device)
    return model

def getBatchLabels(images_batch, error = 0.01):
    """
    Get batch-level hopper level classification labels for a batch of masks.

    Args:
        images_batch (Tensor): Batch of binary segmentation masks.
        error (float): Margin of error for thresholding levels.

    Returns:
        Tensor: Integer labels per image [0, 1, 2].
    """
    listOfLabels = []
    for i in images_batch:
        label = getLabel(i, error)
        listOfLabels.append(label)
    return torch.tensor(listOfLabels)

def getLabel(img, error = 0.01):
    """
    Determine hopper level category from binary mask.

    Args:
        img (Tensor or ndarray): Binary image.
        error (float): Error margin for thresholds.

    Returns:
        int: 0 (Low), 1 (Adequate), or 2 (High) level.
    """

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    img_shape = img.shape
    # Count zero and non-zero pixels
    total_pixels = img_shape[0] * img_shape[1]  # Number of pixels in the ROI

    sum_pixels = np.sum(img)
    # Calculate percentages
    pixel_probability = (sum_pixels / total_pixels) * 100
    low_level = 55 + 55 * error
    high_level = 80 - 80 * error
    if pixel_probability >= high_level :
        return 2
    elif pixel_probability <= low_level:
        return 0
    else:
        return 1

if __name__ == "__main__":
    root = "/home/khush/code/"
    img_folder = "grouped"
    mask_folder = "labels"
    bpath = "exp_runs/"

    config = {
    "lr": 1e-06, #8
    "batch_size":8, 
    "rotation_angle": 14.358234476987889,
    "shear_angle": 0.10768166798428605,
    "num_epochs":10,
    "image_size": (720,1280),
    "noise_std": 0.06967821539400093,
    "blocks": 4,
    "roi": (350, 0, 500, 720)
    }

    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    blocks = config["blocks"]
    
    #train and view the ouput :
    if not os.path.exists(bpath):
        os.mkdir(bpath)
    if  torch.cuda.is_available():
        phase_to_str = lambda p: "train" if p == 0 else "test"

        #transforms = transforms.Compose([ transforms.ToTensor(), transforms.Resize(size=(720, 1280)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomPerspective(distortion_scale=0.1, p=1.0)])
        transform = get_transforms(config)
        datasets = [
            Dataset(
                root,
                img_folder,
                mask_folder,
                set_type=set_type,
                transforms=transform,
            )
            for set_type in ["train", "test"]
        ]

        dataloaders = [
            DataLoader(dataset_i, batch_size=batch_size, num_workers=16)
            for dataset_i in datasets
        ]
        model = createDeepLabv3(num_classes=1, layer_blocks_to_train=blocks)
        model = torch.compile(model)
        
        # or use a pretrained model :
        """
        model_path = "train/bestModels/best_mdl_88.pt"
        model_pretrained = get_Pretrained_model(model_path)
        model = model_pretrained
        """
        model.eval()
        
        train(model, dataloaders, config)
    
        #model_output_video(model_path, video_path)
    else:
        print(" -- Not running on Cuda -- ")

