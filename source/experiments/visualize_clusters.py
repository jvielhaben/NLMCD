import torch
import numpy as np
import pandas as pd

# from skimage.transform import resize
from torchvision.transforms import Normalize


from source.data.utils import DimensionTransformer
from source.data.imagenet import imagenet_loader

import itertools
import math
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import colorsys
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings


def draw_patch_frame(img, index, ax, kernel_size=4, patch_size=16, stride=1):
    # Calculate the number of patches in the image
    n_patches_x = (img.shape[1] // patch_size) // stride - kernel_size + 1
    patch_indices = np.arange(n_patches_x**2).reshape(n_patches_x, n_patches_x)

    # Calculate the row and column of the patch
    row, col = np.where(index == patch_indices)

    # Calculate the top-left corner of the patch in the original image
    top_left_x = col * stride * patch_size
    top_left_y = row * stride * patch_size

    ax.imshow(img)

    # Create a rectangle patch (kernel size)
    rect = patches.Rectangle(
        (top_left_x, top_left_y),
        patch_size * kernel_size,
        patch_size * kernel_size,
        linewidth=0.5,
        edgecolor="yellow",
        facecolor="none",
    )

    # Add the rectangle to the plot
    ax.add_patch(rect)


def show_train_patches(
    cluster_idx,
    soft_clustering,
    hard_clustering,
    token_idx,
    sample_idx,
    cfg_data,
    n_patches,
    ref_cluster_idx=None,
    select_from_all_samples=False,
    n_samples=50,
    random=False,
    title=True,
    ax=None,
):
    if ref_cluster_idx is not None:
        mask = np.logical_and(
            hard_clustering == cluster_idx,
            soft_clustering.argmax(axis=1) == ref_cluster_idx,
        )
    else:
        mask = hard_clustering == cluster_idx
    if select_from_all_samples:
        mask = np.ones(soft_clustering.shape[0]) == 1
    soft_clustering = soft_clustering[mask]
    if random:
        idx = np.random.choice(soft_clustering.shape[0], size=n_samples, replace=False)
    else:
        idx = soft_clustering.max(axis=1).argsort()[-n_samples:]
    if select_from_all_samples:
        idx = soft_clustering[:, cluster_idx].argsort()[-n_samples:]

    sample_idx = sample_idx[mask][idx]
    # token_idx = token_idx[idx]

    # load inout images
    loader = imagenet_loader(
        cfg_data,
        batch_size=n_samples,
        train=True,
        return_label=False,
        cuda=False,
        indices_subsample=sample_idx,
    )
    input_images, y = next(iter(loader))
    y = y.numpy()
    # undo normalization
    norm = loader.dataset.dataset.transform.transforms[-1]
    norm_inverse = Normalize(mean=-norm.mean / norm.std, std=1 / norm.std)
    input_images = norm_inverse(input_images).permute(0, 2, 3, 1).float()
    input_images = torch.clip(input_images, 0, 1)

    if ax is None:
        fig, ax = plt.subplots(n_samples // 10, 10, figsize=(20, 10))
        ax = ax.flatten()

    kernel_size = cfg_data.params.kernel_size if cfg_data.params.pool_token else 1

    for i, idx_i in enumerate(idx):
        if token_idx is None:
            ax[i].imshow(input_images[i])
        else:
            draw_patch_frame(
                input_images[i], token_idx[idx_i], ax[i], kernel_size=kernel_size
            )
        if title:
            ax[i].set_title(
                f"{soft_clustering[idx_i].max():.3f}, {soft_clustering[idx_i].argmax()}",
                fontsize=10,
            )
        ax[i].axis("off")
