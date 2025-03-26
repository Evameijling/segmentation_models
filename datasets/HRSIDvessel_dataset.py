import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rasterio
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

class VesselDetectHRSIDDataset(Dataset):
    def __init__(
        self,
        mode: Literal["train", "valid", "test"],
        use_sen1: bool,
        dataset_dir: str = "/projects/0/prjs1235/FMforSAR/data/HRSIDSARships",
        csv_dir: str = "splits",
        sen1_dir: str = "sar_1band_jpg",
        label_dir: str = "masks",
    ):
        assert mode in ["train", "valid", "test"], f"Invalid mode: {mode}"

        self.mode = mode

        self.use_sen1 = use_sen1

        self.sen1_dir = os.path.join(dataset_dir, sen1_dir)
        self.label_dir = os.path.join(dataset_dir, label_dir)

        df = pd.read_csv(
            os.path.join(dataset_dir, csv_dir, f"{mode}_data.csv"), header=None
        )

        # Format of sample names: {country}_{chip_id}
        self.sample_names = [fn.replace(".jpg", "") for fn in df[0]]

        # Calculate class weights
        self.class_weights = self.calculate_class_weights()

    def __len__(self):
        return len(self.sample_names)

    def calculate_class_weights(self):
        # Calculate the frequency of each class in the dataset
        class_counts = torch.zeros(2, dtype=torch.int64)  # Ensure class_counts is an integer tensor
        for idx in range(len(self)):
            labels = self.get_labels(idx)
            labels = torch.where(labels == 255, torch.tensor(1), labels)  # Convert 255 to 1
            labels = labels.to(torch.int64)  # Convert labels to integer type
            class_counts += torch.bincount(labels.flatten(), minlength=2)

        class_weights = 1 / class_counts.float()  # Convert to float for division
        class_weights /= class_weights.sum()  # Normalize weights
        return class_weights

    def get_labels(self, idx):
        # Load the mask (e.g., using rasterio or another library)
        mask_path = os.path.join(self.label_dir, f"{self.sample_names[idx]}_mask.png")
        mask = rasterio.open(mask_path).read(1)  # Read the mask as a 2D array (H, W)

        # Ensure the mask has the correct shape [1, H, W]
        mask = np.expand_dims(mask, axis=0)  # Add a channel dimension
        return torch.tensor(mask, dtype=torch.float32)

    def get_sen1_img(self, idx):
        sen1_path = os.path.join(self.sen1_dir, f"{self.sample_names[idx]}.jpg")
        sen1_img = rasterio.open(sen1_path).read()

        # print(f"Shape after reading: {sen1_img.shape}")

        # Replace NaNs with 0s
        sen1_img = np.nan_to_num(sen1_img)
        # print(f"Shape after replacing NaNs: {sen1_img.shape}")

        # Normalize the image
        sen1_img = rearrange(sen1_img, "c h w -> h w c")
        # print(f"Shape after rearranging to (h w c): {sen1_img.shape}")
        # sen1_img = (sen1_img - SENTINEL1_MEAN_GRD) / SENTINEL1_STD_GRD
        # # print(f"Shape after normalization: {sen1_img.shape}")
        sen1_img = rearrange(sen1_img, "h w c -> c h w")
        # print(f"Shape after rearranging back to (c h w): {sen1_img.shape}")

        return torch.tensor(sen1_img, dtype=torch.float32)

    def __getitem__(self, idx):
        # Load the images from each specified modality
        images = self.get_sen1_img(idx)  # (2, 256, 256)

        # Load labels
        masks = self.get_labels(idx)  # (256, 256)

        # # During training, apply random flipping and rotation
        # if self.mode == "train":
        #     # Randomly flip all images in the same way
        #     if np.random.rand() > 0.5:
        #         torch.flip(images, dims=(1,))
        #         masks = torch.flip(masks, dims=(0,))

        #     # Randomly rotate all images in the same way by 0, 90, 180, or 270 degrees
        #     k = np.random.randint(0, 4)
        #     torch.rot90(images, k, dims=(1, 2))
        #     masks = torch.rot90(masks, k, dims=(0, 1))

        return images, masks
