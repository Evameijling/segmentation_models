import numpy as np
from scipy.ndimage import label, binary_dilation, binary_erosion
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt

def heat_to_binary_adaptive(heatmap, peak_frac=0.5, min_distance=5, global_threshold=0.2, dilation_size=8):
    """
    Convert a heatmap back to binary by:
      1. Applying a global threshold to filter out low-intensity regions.
      2. Finding local maxima (one per ship).
      3. Watershed on -heatmap using those maxima as markers.
      4. For each watershed region, threshold at peak_frac * its own max.
      5. Apply morphological operations to refine rectangular shapes.
    """
    # Apply a global threshold to filter out low-intensity regions
    heatmap[heatmap < global_threshold] = 0

    # 1) Find peaks in the heatmap
    peaks = peak_local_max(
        heatmap,
        min_distance=min_distance,
        labels=(heatmap > 0)
    )
    
    # Create a markers array with the same shape as the heatmap
    markers = np.zeros_like(heatmap, dtype=int)
    for idx, (row, col) in enumerate(peaks, start=1):
        markers[row, col] = idx  # Assign a unique marker for each peak

    # 2) Watershed to carve out each ship region
    labels = watershed(-heatmap, markers, mask=(heatmap > 0))
    
    # 3) Threshold per-region
    binary = np.zeros_like(heatmap, dtype=bool)
    for ship_id in np.unique(labels):
        if ship_id == 0: 
            continue
        region = (labels == ship_id)
        peak_val = heatmap[region].max()
        thr = peak_frac * peak_val
        binary[region] = (heatmap[region] >= thr)

    # 4) Morphological operations to refine rectangular shapes
    binary = binary_dilation(binary, structure=np.ones((dilation_size, dilation_size)))
    binary = binary_erosion(binary, structure=np.ones((dilation_size, dilation_size)))

    return binary