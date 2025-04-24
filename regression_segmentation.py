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
from datasets.heatmapsHRSIDvessel_dataset import VesselHeatmapDataset
from dataclasses import dataclass
from losses import BinarySeesawLoss, WeightedsumDiceSeesaw, FocalComboLoss

from heat_to_binary import heat_to_binary_adaptive

import argparse

@dataclass
class Config:
    train_model: bool
    epochs: int
    batch_size: int
    learning_rate: float
    architecture: str
    encoder_name: str
    encoder_weights: str
    use_pretrained: bool
    in_channels: int
    out_classes: int
    loss: str
    dataset: type

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Vessel Detection Model")
    # parser.add_argument("--train_model", action="store_true", default=False,
    #                     help="Flag to determine if the model should be trained, default = False")
    parser.add_argument("--train_model", type=str2bool, default=False,
                    help="Flag to determine if the model should be trained, default = False")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs, default = 10")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training, default = 16")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate for optimization, default = 2e-4")
    parser.add_argument("--architecture", type=str, default="unet",
                        help="Model architecture to use, default = 'unet'")
    parser.add_argument("--encoder_name", type=str, default="resnext50_32x4d",
                        help="Encoder name, default = 'resnext50_32x4d'")
    parser.add_argument("--encoder_weights", type=str, default="imagenet",
                        help="Pretrained encoder weights, default = 'imagenet'")
    parser.add_argument("--use_pretrained", action="store_true", default=True,
                        help="Use pretrained weights, default = True")
    parser.add_argument("--in_channels", type=int, default=1,
                        help="Number of input channels, default = 1")
    parser.add_argument("--out_classes", type=int, default=1,
                        help="Number of output classes, default = 1")
    parser.add_argument("--loss", type=str, default="L2loss",
                        choices=['L2loss', 'L1loss'],
                        help="Loss function to be used, default = 'L2loss'")
    parser.add_argument("--dataset", type=str, default="Heatmap",
                        choices=["Heatmap"],
                        help="default = 'Heatmap'")
    parser.add_argument("--sigma_factor", type=float, default=0.3,
                        help="Sigma factor for Gaussian kernel, default = 0.3")
    parser.add_argument("--weight_exponent", type=float, default=0.4,
                        help="Weight exponent for Gaussian kernel, default = 0.4")
    
    return parser.parse_args()
    
class VesselDetectModel(pl.LightningModule):
    def __init__(self, config, T_MAX, train_dataset, **kwargs):
        super().__init__() 
        self.config = config
        self.T_MAX = T_MAX
        self.train_dataset = train_dataset
        self.model = smp.create_model(
            arch=config.architecture,
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights if config.use_pretrained else None,
            in_channels=config.in_channels,
            classes=config.out_classes,
            **kwargs
        )
        # preprocessing parameters for image
        params = smp.encoders.get_preprocessing_params(config.encoder_name)
        self.register_buffer("std", torch.tensor(params["std"][:config.in_channels]).view(1, config.in_channels, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"][:config.in_channels]).view(1, config.in_channels, 1, 1))

        if config.loss == "L2loss":
            self.loss_fn = torch.nn.MSELoss()
        elif config.loss == "L1loss":
            self.loss_fn = torch.nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {config.loss}")
        
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
        image, heatmap = batch
        pred_heatmap = self.forward(image)
        loss = self.loss_fn(pred_heatmap, heatmap)
        mae = F.l1_loss(pred_heatmap, heatmap).item()
        mse = F.mse_loss(pred_heatmap, heatmap).item()
        rmse = torch.sqrt(F.mse_loss(pred_heatmap, heatmap)).item() 

        with torch.no_grad():
            pred_flat = pred_heatmap.view(pred_heatmap.size(0), -1)
            heatmap_flat = heatmap.view(heatmap.size(0), -1)
            corr = [
                torch.corrcoef(torch.stack([p, g]))[0, 1].item()
                for p, g in zip(pred_flat, heatmap_flat)
            ]
            corr_mean = np.mean(corr)

            ss_total = torch.sum((heatmap_flat - heatmap_flat.mean(dim=1, keepdim=True)) ** 2, dim=1)
            ss_residual = torch.sum((heatmap_flat - pred_flat) ** 2, dim=1)
            r2 = 1 - (ss_residual / ss_total)
            r2_mean = r2.mean().item()

        return {"loss": loss, "mae": mae, "mse": mse, "rmse": rmse, "pearson_corr": corr_mean, "r2": r2_mean}

    def shared_epoch_end(self, outputs, stage):
        losses = [x["loss"].item() for x in outputs]
        maes = [x["mae"] for x in outputs]
        mses = [x["mse"] for x in outputs]
        rmses = [x["rmse"] for x in outputs]
        pearsons = [x["pearson_corr"] for x in outputs]
        r2s = [x["r2"] for x in outputs]    

        avg_loss = np.mean(losses)
        avg_mae = np.mean(maes)
        avg_mse = np.mean(mses)
        avg_rmse = np.mean(rmses)
        avg_corr = np.mean(pearsons)
        avg_r2 = np.mean(r2s)

        self.log(f"{stage}/loss", avg_loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/mae", avg_mae, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/mse", avg_mse, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/rmse", avg_rmse, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/pearson_corr", avg_corr, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/r2", avg_r2, prog_bar=True, on_epoch=True)

        print(f"Epoch {self.current_epoch + 1}: {stage.capitalize()} Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, MSE: {avg_mse:.4f}, RMSE: {avg_rmse:.4f}, Pearson Corr: {avg_corr:.4f}, R²: {avg_r2:.4f}")

    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch, "train")
        self.training_step_outputs.append(out)
        return out["loss"]

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        out = self.shared_step(batch, "test")
        self.test_step_outputs.append(out)
        return out

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(self.train_dataset) * self.config.epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def main():
    args = parse_args()
    config = Config(
        train_model=args.train_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        architecture=args.architecture,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        use_pretrained=args.use_pretrained,
        in_channels=args.in_channels,
        out_classes=args.out_classes,
        loss=args.loss,
        dataset=VesselHeatmapDataset
    )

    train_dataset = config.dataset(mode="train", use_sen1=True, sigma_factor=args.sigma_factor, weight_exponent=args.weight_exponent)
    val_dataset = config.dataset(mode="valid", use_sen1=True, sigma_factor=args.sigma_factor, weight_exponent=args.weight_exponent)
    test_dataset = config.dataset(mode="test", use_sen1=True, sigma_factor=args.sigma_factor, weight_exponent=args.weight_exponent)

    images, masks = train_dataset[0]
    print("Images shape:", images.shape)
    print("Masks shape:", masks.shape)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # Visualize some samples
    os.makedirs(f"output_heatmap/samples_{config.dataset.__name__}", exist_ok=True)
    for i in range(20):
        image, mask = train_dataset[i]
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image[0].numpy(), cmap="gray")
        plt.title("Channel 1")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(mask.numpy().squeeze(), cmap="magma")
        plt.title("Ground Truth")
        plt.axis("off")
        plt.savefig(f"output_heatmap/samples_{config.dataset.__name__}/sample{i}.png")

    T_MAX = len(train_loader) * config.epochs

    model = VesselDetectModel(
        config=config, 
        T_MAX=T_MAX,
        train_dataset=train_dataset)

    wandb_logger = WandbLogger(project="heatmap_vesseldetection", name=f"regression-{config.loss}-{config.architecture}-{config.encoder_name}")
    
    wandb_logger.log_hyperparams({
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "architecture": config.architecture,
        "encoder_name": config.encoder_name,
        "encoder_weights": config.encoder_weights,
        "use_pretrained": config.use_pretrained,
        "in_channels": config.in_channels,
        "out_classes": config.out_classes,
        "loss": config.loss,
        "sigma_factor": args.sigma_factor,
        "weight_exponent": args.weight_exponent
    })
    
    trainer = pl.Trainer(max_epochs=config.epochs, logger=wandb_logger, log_every_n_steps=1)

    if config.train_model:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        os.makedirs("models_heatmap", exist_ok=True)
        torch.save(model.state_dict(), f"models_heatmap/heatmap_regression_model.pth")
    else:
        model.load_state_dict(torch.load(f"models_heatmap/heatmap_regression_model.pth"))
        model.eval()

    trainer.test(model, dataloaders=test_loader)

    # Visualization of results
    batch = next(iter(test_loader))
    images, heatmaps = batch

    # retrieve original segmentation masks from base dataset
    base_dataset = test_dataset.base  # assumes VesselHeatmapDataset wraps VesselDetectHRSIDDataset
    original_masks = [base_dataset[i][1] for i in range(len(heatmaps))]  # retrieve original segmentation masks

    with torch.no_grad():
        model.eval()
        predictions = model(images)
    pr_masks = predictions

    os.makedirs(f"output_heatmap/predictions_{config.dataset.__name__}", exist_ok=True)
    for idx, (image, gt_heatmap, pr_mask, orig_mask) in enumerate(zip(images, heatmaps, pr_masks, original_masks)):
        if idx <= 4:
            fig, axs = plt.subplots(1, 5, figsize=(20, 4))

            axs[0].imshow(image[0].numpy(), cmap="gray")
            axs[0].set_title("Channel 1")
            axs[0].axis("off")

            axs[1].imshow(orig_mask.squeeze().numpy(), cmap="gray")
            axs[1].set_title("Original Segmentation")
            axs[1].axis("off")

            axs[2].imshow(gt_heatmap.numpy().squeeze(), cmap="magma")
            axs[2].set_title("Ground Truth Heatmap")
            axs[2].axis("off")

            axs[3].imshow(pr_mask.numpy().squeeze(), cmap="magma")
            axs[3].set_title("Predicted Heatmap")
            axs[3].axis("off")

            binary_map = heat_to_binary_adaptive(pr_mask.squeeze().numpy(), peak_frac=0.6)
            axs[4].imshow(binary_map.astype(float), cmap="gray")
            axs[4].set_title("Binary Map (Watershed)")
            axs[4].axis("off")

            plt.tight_layout()
            fig.savefig(f"output_heatmap/predictions_{config.dataset.__name__}/prediction{idx}.png", bbox_inches='tight', dpi=150)
            plt.close(fig)
        else:
            break

if __name__ == "__main__":
    main()
