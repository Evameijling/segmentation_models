import numpy as np
from scipy.ndimage import label, distance_transform_edt
import os
from skimage.draw import polygon

def segmentation_to_heatmap(
    bin_mask: np.ndarray,
    decay_factor: float = 0.6,
    weight_exponent: float = 0.5,
    normalize: bool = True,
    min_contrib: float = 1e-3
) -> np.ndarray:
    """
    Discrete accumulation *starting at each ship’s center*.

    For each ship:
      - Compute amplitude A = area**(-weight_exponent).
      - Compute Euclidean distance D = dist_to_border inside the ship.
      - Quantize D_int = floor(D) → {0,1,…,maxD}.
      - For ring_idx in [0..maxD]:
          ring = (D_int == maxD - ring_idx)
          heatmap[ring] += A * (decay_factor**ring_idx)
      - Stop early when A*(decay_factor**ring_idx) < min_contrib.
    """
    labels, n_comp = label(bin_mask)
    heatmap = np.zeros_like(bin_mask, dtype=float)

    for comp_id in range(1, n_comp+1):
        comp = (labels == comp_id)
        area = comp.sum()
        if area == 0:
            continue

        # amplitude scales inversely with area
        A = area ** (-weight_exponent)

        # distance‐to‐border (0 at edges, peaks at medial axis)
        D = distance_transform_edt(comp)
        D_int = np.floor(D).astype(int)
        maxD = D_int.max()

        # accumulate inward rings
        for ring_idx in range(maxD+1):
            contrib = A * (decay_factor ** ring_idx)
            if contrib < min_contrib:
                break
            level = maxD - ring_idx
            ring_mask = (D_int == level)
            heatmap[ring_mask] += contrib

    # optional [0,1] rescale
    if normalize:
        vmin, vmax = heatmap.min(), heatmap.max()
        if vmax > vmin:
            heatmap = (heatmap - vmin) / (vmax - vmin)

    return heatmap

# Example demo
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # build a toy mask
    mask = np.zeros((200,300), int)
    mask[50:70, 60:160] = 1
    mask[120:130,200:220] = 1
    # add a rotated rectangle
    angle = np.deg2rad(30)
    L, W, cx, cy = 50, 10, 100, 100
    dx, dy = L/2, W/2
    corners = np.array([[-dx,-dy],[ dx,-dy],[ dx, dy],[-dx, dy]])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    rc = corners.dot(R.T)
    ys, xs = rc[:,1]+cy, rc[:,0]+cx
    rr, cc = polygon(ys, xs, mask.shape)
    mask[rr, cc] = 1

    heat = segmentation_to_heatmap(
        mask,
        decay_factor=0.9,
        weight_exponent=0.4,
        normalize=True,
        min_contrib=1e-4
    )

    # show
    fig, (ax0, ax1) = plt.subplots(1,2,figsize=(8,4))
    ax0.imshow(mask, cmap='gray');   ax0.set_title("Binary mask"); ax0.axis('off')
    ax1.imshow(heat, cmap='magma');  ax1.set_title("Center‑first rings"); ax1.axis('off')

    # Save the figure
    output_dir = "/home/egmelich/segmentation_models/discreteaccumulation_heatmaps"
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap.png"), dpi=300)
