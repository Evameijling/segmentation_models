import torch
from torch.utils.data import Dataset
from datasets.HRSIDvessel_dataset import VesselDetectHRSIDDataset
from gaussiankernel_segmap_to_heatmap import segmentation_to_heatmap  
import numpy as np

class VesselHeatmapDataset(Dataset):
    def __init__(self, *args, sigma_factor=0.3, weight_exponent=0.4, **kwargs):
        super().__init__()
        self.base = VesselDetectHRSIDDataset(*args, **kwargs)
        self.sigma = sigma_factor
        self.wexp  = weight_exponent

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask = self.base[idx]               # mask: 1×H×W or H×W
        # convert to H×W numpy
        m_np = mask.squeeze().cpu().numpy().astype(np.uint8)

        # make heatmap
        h_np = segmentation_to_heatmap(
            m_np,
            sigma_factor=self.sigma,
            weight_exponent=self.wexp,
            normalize=True
        )

        # back to torch tensor (1×H×W), float32
        heat = torch.from_numpy(h_np[None]).float()

        # image is probably already a FloatTensor C×H×W
        return img, heat
