import os

import matplotlib.pyplot as plt
import numpy as np
from datasets.HRSIDvessel_dataset import VesselDetectHRSIDDataset
from scipy.ndimage import label
from skimage.measure import regionprops
from torch.utils.data import DataLoader
from skimage.draw import polygon


def mask_to_heatmap(mask, A_base=1.0, sigma_base=10.0, alpha=1.0, beta=0.5, ref_area=None):
    """
    Convert a binary segmentation mask to a continuous heatmap using instance-specific Gaussian kernels.
    
    Parameters:
    - mask: 2D numpy array of shape (H, W), with binary values (0 background, 1 ship)
    - A_base: Base amplitude for Gaussian
    - sigma_base: Base standard deviation for Gaussian
    - alpha: Exponent for amplitude scaling (small ships ↑peak)
    - beta: Exponent for sigma scaling (small ships ↓spread)
    - ref_area: Reference area; if None, uses median area of all instances
    
    Returns:
    - heatmap: 2D numpy array of floats, same shape as mask
    """
    # Label connected components
    labeled, _ = label(mask)
    props = regionprops(labeled)
    
    # Compute reference area (median)
    if ref_area is None:
        areas = [r.area for r in props]
        print(f"Areas of detected regions: {areas}")
        ref_area = np.median(areas) if areas else 1
    
    heatmap = np.zeros_like(mask, dtype=float)
    H, W = mask.shape
    
    for region in props:
        S = region.area
        y_c, x_c = region.centroid
        
        # Instance-specific parameters
        A = A_base * (ref_area / S) ** alpha
        sigma = sigma_base * (S / ref_area) ** beta
        
        # Bounding box + margin (3σ)
        minr, minc, maxr, maxc = region.bbox
        margin = int(np.ceil(3 * sigma))
        y0, y1 = max(minr - margin, 0), min(maxr + margin, H)
        x0, x1 = max(minc - margin, 0), min(maxc + margin, W)
        
        # Generate Gaussian over the subregion
        ys = np.arange(y0, y1)
        xs = np.arange(x0, x1)
        yv, xv = np.meshgrid(ys, xs, indexing='ij')
        g = A * np.exp(-((xv - x_c)**2 + (yv - y_c)**2) / (2 * sigma**2))
        
        # Accumulate into global heatmap
        heatmap[y0:y1, x0:x1] += g
    
    # Normalize to [0, 1]
    heatmap = np.clip(heatmap, 0, None)
    max_val = heatmap.max()
    if max_val > 0:
        heatmap /= max_val
    
    return heatmap

# Example usage
if __name__ == "__main__":    
    # # Initialize the dataset
    # dataset = VesselDetectHRSIDDataset(
    #     mode="train",  # Choose "train", "valid", or "test"
    #     use_sen1=True,  # Set to True or False based on your requirement
    #     dataset_dir="/projects/0/prjs1235/FMforSAR/data/HRSIDSARships"  # Update if needed
    # )
    
    # # Get one image-mask pair from the dataset
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # image, mask = next(iter(dataloader))
    # print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
    # mask = mask.squeeze().numpy()  # Remove batch dimension

    # Example binary mask
    mask = np.zeros((200, 300), dtype=int)
    mask[50:70, 60:160] = 1   # large ship
    mask[120:130, 200:220] = 1  # small ship
    # rotated ship (30°)
    angle = np.deg2rad(30)
    length, width = 50, 10
    cx, cy = 100, 100
    dx, dy = length/2, width/2
    # rectangle corners centered at (0,0)
    corners = np.array([[-dx, -dy],
                        [ dx, -dy],
                        [ dx,  dy],
                        [-dx,  dy]])
    # rotation matrix
    R = np.array([[ np.cos(angle), -np.sin(angle)],
                  [ np.sin(angle),  np.cos(angle)]])
    rot_corners = corners.dot(R.T)
    xs = rot_corners[:,0] + cx
    ys = rot_corners[:,1] + cy
    rr, cc = polygon(ys, xs, mask.shape)
    mask[rr, cc] = 1

    # Generate heatmap
    heatmap = mask_to_heatmap(mask, sigma_base=15, alpha=1.2, beta=0.7)
    
    # Display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(mask, cmap='gray')
    ax1.set_title('Binary Segmentation Mask')
    ax1.axis('off')
    
    ax2.imshow(heatmap, cmap='hot')
    ax2.set_title('Generated Heatmap')
    ax2.axis('off')
    
    # Save the figure
    output_dir = "/home/egmelich/segmentation_models/isotropic_heatmaps"
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap.png"), dpi=300)







