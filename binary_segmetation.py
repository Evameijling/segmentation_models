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
from losses import BinarySeesawLoss 

@dataclass
class Config:
    train_model: bool = True
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 2e-4
    architecture: str = "unet"
    encoder_name: str = "resnext50_32x4d"
    encoder_weights: str = "imagenet"
    use_pretrained: bool = True
    in_channels: int = 2
    out_classes: int = 1
    loss: str = "seesaw" # Options: dice, focal, seesaw, 
    dataset: type = VesselDetectHRSIDDataset # Options: VesselDetectHRSIDDataset, VesselDetectGRDDataset

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
os.makedirs(f"output/samples_{config.dataset.__name__}", exist_ok=True)
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

    plt.savefig(f"output/samples_{config.dataset.__name__}/sample{i}.png")

class VesselDetectModel(pl.LightningModule):
    def __init__(self, arch,  encoder_name, encoder_weights, use_pretrained, in_channels, out_classes, loss, **kwargs):
        super().__init__() 
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights if use_pretrained else None,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs
        )
        # preprocessing parameters for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"][:2]).view(1, 2, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"][:2]).view(1, 2, 1, 1))

        # for image segmentation dice loss could be the best first choice
        if loss == "dice":
            self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif loss == "focal":
            self.loss_fn = smp.losses.FocalLoss(smp.losses.BINARY_MODE)
        elif loss == "seesaw":
            criterion = BinarySeesawLoss(p=0.8, q=2.0)
            self.loss_fn = criterion
        else:
            raise ValueError(f"Unsupported loss function: {loss}")

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
    
    def compute_siou(self, tp, fp, fn, tn, iou, gamma=0.5, kappa=torch.sqrt(torch.tensor(2.0))):
        total_pixels = tp + fp + fn + tn
        A = tp + fn # ground truth: tp = correct predictions, fn = missed positives 
        B = tp + fp # predictions: tp = correct predictions, fp = false positives
        s = A + B
        s_norm = s / total_pixels
        p_val = 1 - gamma * torch.exp(-torch.sqrt(s_norm/(2*kappa)))
        siou = iou ** p_val
        return siou

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU: first calculate IoU score for each image and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU: aggregate intersection and union over all images and then compute IoU score (most common)
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        # Compute aggregated totals for SIoU
        tp_total = tp.sum()
        fp_total = fp.sum()
        fn_total = fn.sum()
        tn_total = tn.sum()        

        # Calculate SIoU for the overall dataset IoU.
        dataset_siou = self.compute_siou(tp_total, fp_total, fn_total, tn_total, dataset_iou)

         # IoU for the vessel class (binary IoU)
        vessel_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")[1]  # Index 1 corresponds to the vessel class

        # Optionally, calculate SIoU for the vessel IoU
        vessel_siou = self.compute_siou(tp_total, fp_total, fn_total, tn_total, vessel_iou)

        # Mean Accuracy (mAcc): mean of pixel-wise accuracy across all images
        mAcc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")

        # Mean F1 score (mF1): mean of F1 scores across all images
        mF1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        # F1 score for the vessel class
        vessel_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none")[1]  # Index 1 corresponds to the vessel class
        
        # Log metrics
        metrics = {
            f"{stage}/per_image_iou": per_image_iou,
            f"{stage}/dataset_iou": dataset_iou,
            f"{stage}/dataset_siou": dataset_siou,
            f"{stage}/vessel_iou": vessel_iou,
            f"{stage}/vessel_siou": vessel_siou,    
            f"{stage}/mAcc": mAcc,
            f"{stage}/mF1": mF1,
            f"{stage}/F1_vessel": vessel_f1,
        }

        self.log_dict(metrics, prog_bar=True)

        # Log metrics individually
        self.log(f"{stage}/per_image_iou", per_image_iou, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/dataset_iou", dataset_iou, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/dataset_siou", dataset_siou, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/vessel_iou", vessel_iou, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/vessel_siou", vessel_siou, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/mAcc", mAcc, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/mF1", mF1, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/F1_vessel", vessel_f1, prog_bar=True, on_epoch=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        self.log("train_loss", train_loss_info["loss"], prog_bar=True, on_step=True, on_epoch=True)
        return train_loss_info
    
    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear() # empty set output list

        train_metrics = self.trainer.callback_metrics
        print(f"Epoch {self.current_epoch + 1}:")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Train Mean Accuracy (mAcc): {train_metrics['train/mAcc']:.4f}")
        print(f"  Train Per Image IoU: {train_metrics['train/per_image_iou']:.4f}")
        print(f"  Train Dataset IoU: {train_metrics['train/dataset_iou']:.4f}")
        print(f"  Train Dataset SIoU: {train_metrics['train/dataset_siou']:.4f}")
        print(f"  Train Vessel IoU: {train_metrics['train/vessel_iou']:.4f}")
        print(f"  Train Vessel SIoU: {train_metrics['train/vessel_siou']:.4f}")
        print(f"  Train Mean F1 Score (mF1): {train_metrics['train/mF1']:.4f}")
        print(f"  Train F1 Score (Vessel): {train_metrics['train/F1_vessel']:.4f}")
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        self.log("valid_loss", valid_loss_info["loss"], prog_bar=True, on_step=False, on_epoch=True)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()  # empty set output list

        valid_metrics = self.trainer.callback_metrics  # Access logged metrics
        print(f"Epoch {self.current_epoch + 1}:")
        print(f"  Validation Loss: {valid_metrics['valid_loss']:.4f}")
        print(f"  Validation Mean Accuracy (mAcc): {valid_metrics['valid/mAcc']:.4f}")
        print(f"  Validation Per Image IoU: {valid_metrics['valid/per_image_iou']:.4f}")
        print(f"  Validation Dataset IoU: {valid_metrics['valid/dataset_iou']:.4f}")
        print(f"  Validation Dataset SIoU: {valid_metrics['valid/dataset_siou']:.4f}")
        print(f"  Validation Vessel IoU: {valid_metrics['valid/vessel_iou']:.4f}")
        print(f"  Validation Vessel SIoU: {valid_metrics['valid/vessel_siou']:.4f}")
        print(f"  Validation Mean F1 Score (mF1): {valid_metrics['valid/mF1']:.4f}")
        print(f"  Validation F1 Score (Vessel): {valid_metrics['valid/F1_vessel']:.4f}")
        return

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear() # empty set output list

        # Print test metrics
        test_metrics = self.trainer.callback_metrics
        print(f"""
        Test Metrics:
        - Mean Accuracy (mAcc): {test_metrics['test/mAcc']:.4f}
        - Per image IoU: {test_metrics['test/per_image_iou']:.4f}
        - Dataset IoU: {test_metrics['test/dataset_iou']:.4f}
        - Dataset SIoU: {test_metrics['test/dataset_siou']:.4f}
        - Vessel IoU: {test_metrics['test/vessel_iou']:.4f}
        - Vessel SIoU: {test_metrics['test/vessel_siou']:.4f}
        - Mean F1 Score (mF1): {test_metrics['test/mF1']:.4f}
        - F1 Score (Vessel): {test_metrics['test/F1_vessel']:.4f}
        """)
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

model = VesselDetectModel(
    config.architecture, 
    config.encoder_name, 
    config.encoder_weights, 
    config.use_pretrained, 
    config.in_channels,
    config.out_classes,
    config.loss)

pretrained_status = "pretrained" if config.use_pretrained else "scratch"
wandb_logger = WandbLogger(project="loss-unet-vessel-detection", name=f"{config.loss}-{config.dataset.__name__}-{config.architecture}-{config.encoder_name}-{pretrained_status}")

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
    # Ensure the 'models' directory exists
    os.makedirs("models", exist_ok=True)
    
    # Save the trained model
    pretrained_status = "pretrained" if config.use_pretrained else "scratch"
    torch.save(model.state_dict(), f"models/vessel_detect_model_{config.loss}-{config.dataset.__name__}-{config.architecture}-{config.encoder_name}-{pretrained_status}.pth")
else:
    # Load the saved model
    pretrained_status = "pretrained" if config.use_pretrained else "scratch"
    model.load_state_dict(torch.load(f"models/vessel_detect_model_{config.loss}-{config.dataset.__name__}-{config.architecture}-{config.encoder_name}-{pretrained_status}.pth"))
    model.eval()  # Set the model to evaluation mode

# run validation dataset
valid_metrics = trainer.validate(model, dataloaders=val_loader, verbose=False)

# run test dataset
test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)

# result visualization
batch = next(iter(test_loader))
images, masks = batch
with torch.no_grad():
    model.eval()
    logits = model(images)
pr_masks = logits.sigmoid()
os.makedirs(f"output/predictions_{config.dataset.__name__}", exist_ok=True)
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

        plt.savefig(f"output/predictions_{config.dataset.__name__}/prediction{idx}.png")
    else:
        break

