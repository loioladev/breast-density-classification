import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch.utils.data import DataLoader

logger = logging.getLogger()


# Inspired from https://towardsdatascience.com/demystifying-pytorchs-weightedrandomsampler-by-example-a68aceccb452
def visualize_dataloader(dataloader: DataLoader, id_to_label: dict, path: str) -> None:
    """
    Visualize the images of a batch

    :param dataloader: The dataloader to visualize
    :param id_to_label: The mapping of the id to the label
    :param path: The path to save the plot
    """
    images, labels = next(iter(dataloader))
        
    # -- plot the first batch of images
    columns = 4
    rows = math.ceil(len(images) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3))
    axes = axes.flatten()

    if len(images) == 1:
        axes = [axes]

    for idx, image in enumerate(images):
        image = image.permute(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())
        axes[idx].imshow(image)
        axes[idx].axis("off")

    plt.subplots_adjust(wspace=0.05, hspace=0.02)
    plt.savefig(path + "/dataloader_images.png")
    plt.close()
    logger.info(f"Batch visualization saved at {path}")


def plot_confusion_matrix(cm, path: str) -> None:
    """
    Plot the confusion matrix

    :param cm: The confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(Path(path) / "confusion_matrix.png")


def plot_roc_curve(roc_data, auroc):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(
        roc_data["fpr"],
        roc_data["tpr"],
        color="darkorange",
        lw=2,
        label=f"ROC curve (AUROC = {auroc:.3f})",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()
