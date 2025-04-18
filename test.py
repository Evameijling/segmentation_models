import numpy as np
from scipy.ndimage import label, distance_transform_edt, gaussian_filter
import os

def segmentation_to_heatmap(
    bin_mask: np.ndarray,
    sigma_factor: float = 0.02,
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
        
        # 2c) Distance‐to‐edge map (peaks at component center)
        dist_map = distance_transform_edt(comp_mask)
        # Avoid divide‐by‐zero
        if dist_map.max() > 0:
            dist_map = dist_map / dist_map.max()
        else:
            dist_map = dist_map
        
        # 2d) Gaussian blur of the binary component
        blurred = gaussian_filter(comp_mask.astype(float), sigma=sigma, mode='constant')
        
        # 2e) Combine: amplitude * blur * center‐weight
        heatmap += weight * blurred * dist_map

    # 3) Optional normalization
    if normalize:
        vmin, vmax = heatmap.min(), heatmap.max()
        if vmax > vmin:
            heatmap = (heatmap - vmin) / (vmax - vmin)
    
    return heatmap

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # create a toy mask: two rectangles of different sizes
    mask = np.zeros((200, 200), dtype=bool)
    mask[20:60, 30:80] = True      # small vessel
    mask[100:180, 50:150] = True   # large vessel

    heat = segmentation_to_heatmap(mask,
                                   sigma_factor=0.015,
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
    output_dir = "/home/egmelich/segmentation_models/test"
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap.png"), dpi=300)
