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
from pytorch_lightning.loggers import WandbLogger

from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset

from datasets.GRDvessel_dataset import VesselDetectGRDDataset
from datasets.HRSIDvessel_dataset import VesselDetectHRSIDDataset
from dataclasses import dataclass

@dataclass
class Config:
    train_model: bool = True
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 2e-4
    architecture: str = "unet"
    encoder_name: str = "resnext50_32x4d"
    in_channels: int = 2
    out_classes: int = 1
    dataset: type = VesselDetectGRDDataset # VesselDetectHRSIDDataset or VesselDetectGRDDataset

config = Config()

train_dataset = config.dataset(mode="train", use_sen1=True)
if config.dataset == VesselDetectGRDDataset:
    val_dataset = config.dataset(mode="validation", use_sen1=True)
elif config.dataset == VesselDetectHRSIDDataset:
    val_dataset = config.dataset(mode="valid", use_sen1=True)
else:
    raise ValueError("Unsupported dataset type in config.dataset")
test_dataset = config.dataset(mode="test", use_sen1=True)

images, masks = train_dataset[0]
print("Images shape:", images.shape)
print("Masks shape:", masks.shape)

# It is a good practice to check datasets don`t intersects with each other
assert set(test_dataset.sample_names).isdisjoint(set(train_dataset.sample_names))
assert set(test_dataset.sample_names).isdisjoint(set(val_dataset.sample_names))
assert set(train_dataset.sample_names).isdisjoint(set(val_dataset.sample_names))

print("Length of training dataset:", len(train_dataset))
print("Length of validation dataset:", len(val_dataset))
print("Length of test dataset:", len(test_dataset))

# n_cpu = os.cpu_count()
n_cpu = 4

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=n_cpu
)
val_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=n_cpu
)
test_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=n_cpu
)

T_MAX = config.batch_size * len(train_loader)

# Visualize some samples
os.makedirs(f"samples_{config.dataset.__name__}", exist_ok=True)
for i in range(5):
    image, mask = train_dataset[i]
    plt.figure(figsize=(12, 4))  

    # Visualize the first channel (e.g., VV)
    plt.subplot(1, 3, 1)
    plt.imshow(image[0].numpy(), cmap="gray")
    plt.title("Channel 1 (VV)")

    if image.shape[0] > 1:
        # Visualize the second channel (e.g., VH)
        plt.subplot(1, 3, 2)
        plt.imshow(image[1].numpy(), cmap="gray")
        plt.title("Channel 2 (VH)")

    # Visualize the mask
    plt.subplot(1, 3, 3)
    plt.imshow(mask.squeeze().numpy(), cmap="gray")  # Remove the singleton dimension
    plt.title("Mask")

    plt.savefig(f"samples/sample{i}.png")

class VesselDetectModel(pl.LightningModule):
    def __init__(self, arch,  encoder_name, in_channels, out_classes, **kwargs):
        super().__init__() 
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            # encoder_weights="imagenet",
            **kwargs
        )
        # preprocessing parameters for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"][:2]).view(1, 2, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"][:2]).view(1, 2, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, image):
        # normalize image 
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask 
    
    def shared_step(self, batch, stage):
        image, mask = batch

        assert image.ndim == 4, "Input image must have 4 dimensions [batch_size, num_channels, height, width]"

        h,w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0, "Image height and width must be divisible by 32"

        assert mask.ndim == 4, "Input mask must have 4 dimensions [batch_size, num_classes, height, width]"

        # forward pass
        logits_mask = self.forward(image)

        # calculate loss
        loss = self.loss_fn(logits_mask, mask)

        # convert logits to probabilities
        prob_mask = logits_mask.sigmoid()
        # apply thresholding for classification (>0.5 probability is considered as a vessel)
        pred_mask = (prob_mask > 0.5).float()

        # compute true positive, false positive, true negative, false negative
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    
    def shared_epoch_end(self, outputs, stage):
        # aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU: first calculate IoU score for each image and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        # dataset IoU: aggregate intersection and union over all images and then compute IoU score
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        
        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

        # Log metrics individually
        self.log(f"{stage}_per_image_iou", per_image_iou, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_dataset_iou", dataset_iou, prog_bar=True, on_epoch=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        self.log("train_loss", train_loss_info["loss"], prog_bar=True, on_step=True, on_epoch=True)
        return train_loss_info
    
    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear() # empty set output list
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        self.log("valid_loss", valid_loss_info["loss"], prog_bar=True, on_step=False, on_epoch=True)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()  # empty set output list
        return

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear() # empty set output list
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
        return

model = VesselDetectModel(config.architecture, config.encoder_name, in_channels=2, out_classes=1)

wandb_logger = WandbLogger(project="unet-vessel-detection", name=f"{config.dataset.__name__}-{config.architecture}-{config.encoder_name}")

wandb_logger.log_hyperparams({
    "epochs": config.epochs,
    "learning_rate": config.learning_rate,
    "batch_size": config.batch_size,
    "model_architecture": config.architecture,
    "encoder_name": config.encoder_name,
    "in_channels": config.in_channels,
    "out_classes": config.out_classes
})

trainer = pl.Trainer(max_epochs=config.epochs, log_every_n_steps=1, logger=wandb_logger)

if config.train_model:
    trainer.fit(
        model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    # Save the trained model
    torch.save(model.state_dict(), "vessel_detect_model.pth")
else:
    # Load the saved model
    model.load_state_dict(torch.load("vessel_detect_model.pth"))
    model.eval()  # Set the model to evaluation mode

# run validation dataset
valid_metrics = trainer.validate(model, dataloaders=val_loader, verbose=False)

# run test dataset
test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)

print(f"""
Validation Metrics:
- Loss: {valid_metrics[0]["valid_loss"]}
- Per image IoU: {valid_metrics[0]["valid_per_image_iou"]}
- Dataset IoU: {valid_metrics[0]["valid_dataset_iou"]}

Test Metrics:
- Per image IoU: {test_metrics[0]["test_per_image_iou"]}
- Dataset IoU: {test_metrics[0]["test_dataset_iou"]}
""")

# result visualization
batch = next(iter(test_loader))
images, masks = batch
with torch.no_grad():
    model.eval()
    logits = model(images)
pr_masks = logits.sigmoid()
os.makedirs(f"predictions_{config.dataset.__name__}", exist_ok=True)
for idx, (image, gt_mask, pr_mask) in enumerate(
    zip(images, masks, pr_masks)
):
    if idx <= 4:
        plt.figure(figsize=(15, 5))

        # Visualize the first channel (e.g., VV)
        plt.subplot(1, 4, 1)
        plt.imshow(image[0].numpy(), cmap="gray")
        plt.title("Channel 1 (VV)")
        plt.axis("off")

        if image.shape[0] > 1:
            # Visualize the second channel (e.g., VH)
            plt.subplot(1, 4, 2)
            plt.imshow(image[1].numpy(), cmap="gray")
            plt.title("Channel 2 (VH)")
            plt.axis("off")

        # Visualize the ground truth mask
        plt.subplot(1, 4, 3)
        plt.imshow(gt_mask.numpy().squeeze(), cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        # Visualize the predicted mask
        plt.subplot(1, 4, 4)
        plt.imshow(pr_mask.numpy().squeeze(), cmap="gray")
        plt.title("Prediction")
        plt.axis("off")

        plt.savefig(f"predictions/prediction{idx}.png")
    else:
        break

