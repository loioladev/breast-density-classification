import logging
import math

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import seaborn as sns

logger = logging.getLogger()


# Inspired from https://towardsdatascience.com/demystifying-pytorchs-weightedrandomsampler-by-example-a68aceccb452
def visualize_dataloader(dataloader: DataLoader, id_to_label: dict, path: str) -> None:
    """
    Visualize the batch distribution of the dataloader in a bar plot

    :param dataloader: The dataloader to visualize
    :param id_to_label: The mapping of the id to the label
    :param path: The path to save the plot
    """
    total_classes = len(id_to_label)
    class_batch_counts = {class_id: [] for class_id in range(total_classes)}
    images = []
    for idx, batch in enumerate(dataloader):
        # -- obtain the classes of the batch
        if idx == 0:
            images = batch[0]
        classes = batch[1].tolist()

        # -- count the number of classes in the batch
        for class_id in range(total_classes):
            class_batch_counts[class_id].append(classes.count(class_id))

    # -- plot the distribution in a clustered bar plot
    fig, ax = plt.subplots(figsize=(12, 8))

    bar_width = 0.2
    for class_id, counts in class_batch_counts.items():
        ax.bar(
            np.arange(len(counts)) + class_id * bar_width,
            counts,
            bar_width,
            label=id_to_label[class_id],
        )
        ax.set_xticks(np.arange(len(counts)) + bar_width / 2)

    ax.set_xlabel("Batch")
    ax.set_ylabel("Count")
    ax.legend()
    plt.savefig(path + "/dataloader_distribution.png")
    plt.close()
    logger.info(f"Distribution of data visualization saved at {path}")

    # -- plot the first batch of images
    columns = 4
    rows = math.ceil(len(images) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3))
    axes = axes.flatten()

    if len(images) == 1:
        axes = [axes]

    for idx, image in enumerate(images):
        axes[idx].imshow(image.permute(1, 2, 0))
        axes[idx].axis("off")

    plt.subplots_adjust(wspace=0.05, hspace=0.02)
    plt.savefig(path + "/dataloader_images.png")
    plt.close()
    logger.info(f"Batch visualization saved at {path}")


def plot_confusion_matrix(cm) -> None:
    """
    Plot the confusion matrix

    :param cm: The confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def plot_roc_curve(roc_data, auroc):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(roc_data['fpr'], roc_data['tpr'], color='darkorange', lw=2, 
             label=f'ROC curve (AUROC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
