import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class BinarySeesawLoss(nn.Module):
    """
    Seesaw Loss for binary segmentation (for instance Vessel vs. Background) that dynamically reweights the gradients based on the class occurrence.

    This implementation is a (PyTorch) reformulation of the original Seesaw Loss, which was introduced for long-tailed instance segmentation:
    
    Wang, Jiaqi, et al. "Seesaw Loss for Long-Tailed Instance Segmentation."
    2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 9690–9699.
    doi:10.1109/CVPR46437.2021.00957
    
    Args:
        p (float): Controls the mitigation effect (a higher p increases the mitigation effect, i.e. downweights the loss strongly when one class dominates). Default = 0.8.
        q (float): Controls the compensation effect (a higher q increases penalty when the model is confidently wrong). Default = 2.0.
        temperature (float): Scaling factor for logits. Default = 3.0.
    """
    def __init__(self, p=0.8, q=2.0, temperature=1.0):
        super(BinarySeesawLoss, self).__init__()
        self.p = p
        self.q = q
        self.temperature = temperature
        self.eps = 1e-6

    def forward(self, logits, targets):
        """
        Compute the Binary Seesaw Loss.
        
        Args:
            logits: Model predictions (raw logits) [B, 2, H, W] (for target class & background)
            targets: Ground truth labels [B, H, W], values in {0, 1}
        
        Returns:
            loss (scalar): The computed Seesaw loss.
        """
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        
        # Handle single-channel logits ([B, 1, H, W]) or two-channel logits ([B, 2, H, W])
        if logits.shape[1] == 1:
            logits = torch.cat([-logits, logits], dim=1)

        # Scale logits by temperature
        logits = logits * self.temperature
        
        # Convert targets to one-hot encoding (shape: [B, 2, H, W])
        targets_one_hot = F.one_hot(targets.long(), num_classes=2).permute(0, 3, 1, 2)
        
        # Compute softmax probabilities for target class & background
        exp_logits = torch.exp(logits)
        softmax_denom = exp_logits.sum(dim=1, keepdim=True)
        probs = exp_logits / (softmax_denom + self.eps)
        
        # Extract per-class probabilities
        probs_vessel = probs[:, 1, :, :]  # Target class probability
        probs_background = probs[:, 0, :, :]  # Background class probability
        
        # Compute pixel-wise class counts (sum over spatial dimensions)
        N_v = torch.sum(targets == 1) + self.eps  # Target class pixel count
        N_b = torch.sum(targets == 0) + self.eps  # Background pixel count
        
        # Mitigation Factor (reduce penalty on background when it dominates)
        M_vb = torch.min(torch.tensor(1.0, device=logits.device), (N_b / N_v) ** self.p)
        M_bv = torch.min(torch.tensor(1.0, device=logits.device), (N_v / N_b) ** self.p)
        M_bv = torch.clamp(M_bv, min=0.1)
        
        # Compensation Factor (increase penalty when a class is misclassified)
        # C_vb = torch.max(torch.tensor(1.0, device=logits.device), (probs_background / probs_vessel) ** self.q)
        # C_bv = torch.max(torch.tensor(1.0, device=logits.device), (probs_vessel / probs_background) ** self.q)

        C_vb = torch.max(
            torch.tensor(1.0, device=logits.device), 
            ( (probs_background.detach() / probs_vessel.detach()) ** self.q )
        )
        C_bv = torch.max(
            torch.tensor(1.0, device=logits.device), 
            ( (probs_vessel.detach() / probs_background.detach()) ** self.q )
        )

        C_vb = torch.clamp(C_vb, max=5.0)
        C_bv = torch.clamp(C_bv, max=5.0)
        
        # Compute reweighted probabilities
        seesaw_vessel = M_vb * C_vb
        seesaw_background = M_bv * C_bv
        
        # Compute Seesaw Loss
        loss_vessel = -targets_one_hot[:, 1, :, :] * torch.log(probs_vessel + self.eps) * seesaw_vessel
        loss_background = -targets_one_hot[:, 0, :, :] * torch.log(probs_background + self.eps) * seesaw_background
        loss = loss_vessel + loss_background
        
        return loss.mean()
    

class WeightedsumDiceSeesaw(nn.Module):
    def __init__(self, dice_weight=0.5, seesaw_weight=0.5, p=0.8, q=2.0):
        """
        Initialize the combined loss function.
        Args:
            dice_weight (float): Weight for the Dice Loss.
            seesaw_weight (float): Weight for the Seesaw Loss.
            p (float): Parameter for the Seesaw Loss.
            q (float): Parameter for the Seesaw Loss.
        """
        super().__init__()
        self.dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.seesaw_loss = BinarySeesawLoss(p=p, q=q)
        self.dice_weight = dice_weight
        self.seesaw_weight = seesaw_weight

    def forward(self, logits, targets):
        """
        Compute the combined loss.
        Args:
            logits (torch.Tensor): Predicted logits.
            targets (torch.Tensor): Ground truth masks.
        Returns:
            torch.Tensor: Combined loss value.
        """
        dice = self.dice_loss(logits, targets)
        seesaw = self.seesaw_loss(logits, targets)
        combined_loss = self.dice_weight * dice + self.seesaw_weight * seesaw
        return combined_loss


class FocalComboLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, beta=1.0, weight=None):
        """
        Initialize the Focal Combo Loss.
        Args:
            alpha (float): Weighting factor for Focal Loss and Dice Loss combination. Default is 0.5.
            gamma (float): Focusing parameter for Focal Loss. Default is 2.0.
            beta (float): Parameter for the Dice/F1-based loss. Default is 1.0.
            weight (torch.Tensor): Class weights for Focal Loss. Default is None.
        """
        super(FocalComboLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.weight = weight

    def forward(self, logits, targets):
        """
        Compute the Focal Combo Loss.
        Args:
            logits (torch.Tensor): Predicted logits [B, 1, H, W].
            targets (torch.Tensor): Ground truth labels [B, H, W] or [B, 1, H, W].
        Returns:
            torch.Tensor: Combined loss value.
        """
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)  # Remove channel dimension if present

        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)  # pt = P_t (probability for the true class)

        # Focal Loss
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + 1e-6)
        if self.weight is not None:
            focal_loss = focal_loss * (self.weight[1] * targets + self.weight[0] * (1 - targets))
        focal_loss = focal_loss.mean()

        # Dice/F1-based Loss
        tp = torch.sum(probs * targets)  # True positives
        fp = torch.sum(probs * (1 - targets))  # False positives
        fn = torch.sum((1 - probs) * targets)  # False negatives
        dice_loss = 1 - (2 * tp / (2 * tp + fp + fn + 1e-6)) ** (1 / self.beta)

        # Combine Focal Loss and Dice Loss
        combined_loss = self.alpha * focal_loss + (1 - self.alpha) * dice_loss
        return combined_loss