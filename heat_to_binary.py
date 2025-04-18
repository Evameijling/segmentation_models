import numpy as np
from scipy.ndimage import label
from skimage.feature    import peak_local_max
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt

def heat_to_binary_adaptive(heatmap, peak_frac=0.3, min_distance=5):
    """
    Convert a heatmap back to binary by:
      1. Finding local maxima (one per ship). 
      2. Watershed on -heatmap using those maxima as markers.
      3. For each watershed region, threshold at peak_frac * its own max.
    """
    # 1) find peaks in the heatmap
    peaks = peak_local_max(
        heatmap,
        min_distance=min_distance,
        indices=False,
        labels=(heatmap > 0)
    )
    markers, _ = label(peaks)
    
    # 2) watershed to carve out each ship region
    labels = watershed(-heatmap, markers, mask=(heatmap > 0))
    
    # 3) threshold per-region
    binary = np.zeros_like(heatmap, dtype=bool)
    for ship_id in np.unique(labels):
        if ship_id == 0: 
            continue
        region = (labels == ship_id)
        peak_val = heatmap[region].max()
        thr = peak_frac * peak_val
        binary[region] = (heatmap[region] >= thr)
    return binary
