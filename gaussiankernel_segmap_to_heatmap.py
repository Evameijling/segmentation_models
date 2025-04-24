import numpy as np
from scipy.ndimage import label, distance_transform_edt, gaussian_filter
import os
from skimage.draw import polygon
import matplotlib.pyplot as plt
from datasets.HRSIDvessel_dataset import VesselDetectHRSIDDataset
from torch.utils.data import DataLoader

def segmentation_to_heatmap(
    bin_mask: np.ndarray,
    sigma_factor: float = 0.1,
    weight_exponent: float = 0.5,
    normalize: bool = True
) -> np.ndarray:
    """
    Convert a binary segmentation mask of ships into a heatmap.
    
    Parameters
    ----------
    bin_mask : np.ndarray, shape (H, W)
        Binary mask (dtype=bool or {0,1}) where ships==1.
    sigma_factor : float
        Controls how blur σ scales with component size:
        σ = sigma_factor * sqrt(component_area).
    weight_exponent : float
        Controls inverse‐size weighting:
        amplitude ∝ (1 / area**weight_exponent).
    normalize : bool
        If True, rescales final heatmap to [0,1].

    Returns
    -------
    heatmap : np.ndarray, shape (H, W), dtype=float
        Floating‐point heatmap.
    """
    # 1) Label each connected ship
    labels, n_comp = label(bin_mask)
    heatmap = np.zeros_like(bin_mask, dtype=float)
    
    # 2) Process each component
    for comp_id in range(1, n_comp+1):
        comp_mask = (labels == comp_id)
        area = comp_mask.sum()
        if area == 0:
            continue
        
        # 2a) Inverse‐size amplitude weight
        weight = area ** (-weight_exponent)
        
        # 2b) σ scales with sqrt(area)
        sigma = sigma_factor * np.sqrt(area)
        
        # 1) build a normalized distance map
        dist_map = distance_transform_edt(comp_mask)
        dist_map /= dist_map.max()

        # 2) blur the *product* of mask and dist_map
        blurred = gaussian_filter((comp_mask.astype(float) * dist_map),
                                sigma=sigma, mode='constant')

        # 3) accumulate
        heatmap += weight * blurred

    # 3) Optional normalization
    if normalize:
        vmin, vmax = heatmap.min(), heatmap.max()
        if vmax > vmin:
            heatmap = (heatmap - vmin) / (vmax - vmin)
    
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

    heat = segmentation_to_heatmap(mask,
                                   sigma_factor=0.3,
                                   weight_exponent=0.4)

    # display
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8,4))
    ax0.imshow(mask, cmap='gray')
    ax0.set_title("Binary mask")
    ax0.axis('off')

    ax1.imshow(heat, cmap='magma')
    ax1.set_title("Ship heatmap")
    ax1.axis('off')

    # Save the figure
    output_dir = "/home/egmelich/segmentation_models/gaussiankernel_heatmaps"
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap.png"), dpi=300)