import os
from datasets.HRSIDvessel_dataset import VesselDetectHRSIDDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def save_image_mask_pairs():
    # Initialize the dataset
    dataset = VesselDetectHRSIDDataset(
        mode="train",  # Choose "train", "valid", or "test"
        use_sen1=True,  # Set to True or False based on your requirement
        dataset_dir="/projects/0/prjs1235/FMforSAR/data/HRSIDSARships"  # Update if needed
    )

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Directory to save the images and masks
    output_dir = "/home/egmelich/segmentation_models/output_pairs"
    os.makedirs(output_dir, exist_ok=True)

    # Save 5 image-mask pairs
    for i, (image, mask) in enumerate(dataloader):
        if i >= 5:  # Stop after saving 5 pairs
            break

        # Convert tensors to numpy arrays
        image_np = image.squeeze().numpy()  # Remove batch dimension
        mask_np = mask.squeeze().numpy()

        # Handle multi-channel images
        if image_np.ndim == 3:  # If the image has channels (C, H, W)
            for channel in range(image_np.shape[0]):
                plt.imsave(
                    os.path.join(output_dir, f"image_{i}_channel_{channel}.png"),
                    image_np[channel],
                    cmap="gray"
                )
        else:  # Single-channel image
            plt.imsave(os.path.join(output_dir, f"image_{i}.png"), image_np, cmap="gray")

        # Save the mask
        plt.imsave(os.path.join(output_dir, f"mask_{i}.png"), mask_np, cmap="gray")

        print(f"Saved image_{i}.png and mask_{i}.png")

    print(f"Saved 5 image-mask pairs in {output_dir}")

if __name__ == "__main__":
    save_image_mask_pairs()