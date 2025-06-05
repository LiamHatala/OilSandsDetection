"""
model.py

Defines a custom DeepLabV3 model based on a pretrained ResNet101 backbone,
with additional trainable blocks added to the classifier for oil sands segmentation.

The model replaces the default classifier head with a configurable number
of Conv-BatchNorm-ReLU blocks, followed optionally by a Softmax classifier
for the final output classes.
"""

from torchvision import models, transforms
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
import torch

def createDeepLabv3(num_classes: int = 1, layer_blocks_to_train: int = 1):
    """
    Creates a DeepLabV3 segmentation model with a custom classifier head.

    The function loads a pretrained DeepLabV3 model (with a ResNet101 backbone),
    freezes the backbone layers, removes the auxiliary classifier, and replaces
    the classifier head with a sequence of Conv2D, BatchNorm, and ReLU layers,
    followed by a final Conv2D and Softmax layer.

    Args:
        num_classes (int, optional): Number of output classes for segmentation. Defaults to 1.
        layer_blocks_to_train (int, optional): Number of custom blocks to add to the classifier. 
                                               Each block includes Conv2D → BatchNorm → ReLU.
                                               The last block outputs `num_classes` channels.

    Returns:
        model (torch.nn.Module): A modified DeepLabV3 model ready for training.
    """
    
    # load the pretrained model with ResNet101 backbone
    model = models.segmentation.deeplabv3_resnet101(
        weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
    )
    layers_except_trainable = list(model.classifier.children())[:-4]

    # freeze training for all layers except new layers
    # also, remove auxiliary classifier
    model.backbone.requires_grad_(False)
    if model.aux_classifier is not None:
        model.aux_classifier = None

    for layer in layers_except_trainable:
        for param in layer.parameters():
            param.requires_grad = False

    # For now, adding alternating conv2d and batch norm layers
    new_layers = []
    for i in range(layer_blocks_to_train):
        # intermediate Conv2d + BatchNorm2d + ReLU
        conv1 = torch.nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        bn_layer = torch.nn.BatchNorm2d(
            256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
        )
        relu = torch.nn.ReLU()

        out_last = num_classes if i == layer_blocks_to_train - 1 else 256
        new_layers.append(conv1)
        new_layers.append(bn_layer)
        new_layers.append(relu)

        if i == layer_blocks_to_train - 1:
            conv2 = torch.nn.Conv2d(
                256, out_last, kernel_size=(1, 1), stride=(1, 1), bias=False
            )
            new_layers.append(conv2)
            new_layers.append(torch.nn.Softmax(dim=1))
    # replace the original classifier with our custom sequence
    model.classifier = torch.nn.Sequential(*layers_except_trainable, *new_layers)

    model.train()
    return model
