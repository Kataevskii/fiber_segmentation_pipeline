import torch
import torch.nn as nn
import torch.nn.functional as F

class IntensityLoss(nn.Module):
    """
    Specialized Loss for IntensityUNet3D:
    Trains the network to predict a continuous 3D Gaussian potential field I(x) in [-1, 1].
    - Foreground-weighted MSE + Smooth L1
    - Continuous Centerline Dice Loss
    - Negative Intersection Dip Loss
    """
    def __init__(self, intensity_weight=1.0, dice_weight=1.0, neg_weight=0.5, eps=1e-8):
        super().__init__()
        self.intensity_weight = intensity_weight
        self.dice_weight = dice_weight
        self.neg_weight = neg_weight
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, pred_intensity, gt_intensity):
        """
        pred_intensity: (B, 1, D, H, W) in [-1, 1]
        gt_intensity:   (B, 1, D, H, W) in [-1, 1]
        """
        # 1. Continuous Gaussian Radial Potential Field Loss
        fg_mask = (torch.abs(gt_intensity) > 0.05).float()
        cl_core = (gt_intensity > 0.30).float()
        voxel_weights = 1.0 + 3.0 * fg_mask + 4.0 * cl_core
        voxel_weight_sum = voxel_weights.sum()

        diff_sq = (pred_intensity - gt_intensity) ** 2
        intensity_mse = (diff_sq * voxel_weights).sum() / voxel_weight_sum

        l1_diff = F.smooth_l1_loss(pred_intensity, gt_intensity, reduction='none')
        intensity_l1 = (l1_diff * voxel_weights).sum() / voxel_weight_sum
        intensity_loss = intensity_mse + intensity_l1

        # 2. Continuous Soft Centerline Dice Loss (Continuous vs Continuous)
        pos_gt = F.relu(gt_intensity)
        pos_pred = F.relu(pred_intensity)

        intersection = (pos_pred * pos_gt).sum()
        dice_denom = pos_pred.pow(2).sum() + pos_gt.pow(2).sum()
        soft_dice = (2.0 * intersection + self.eps) / (dice_denom + self.eps)
        dice_loss = 1.0 - soft_dice

        # Centerline Binary Dice Score for human-readable monitoring (> 0.30)
        bin_pred = (pos_pred > 0.30).float()
        bin_gt = (pos_gt > 0.30).float()
        bin_dice = (2.0 * (bin_pred * bin_gt).sum() + self.eps) / (bin_pred.sum() + bin_gt.sum() + self.eps)

        # 3. Negative Intersection Dip Loss
        neg_gt = F.relu(-gt_intensity)
        neg_pred = F.relu(-pred_intensity)
        neg_intersection_loss = self.mse(neg_pred, neg_gt)

        total_loss = (
            self.intensity_weight * intensity_loss +
            self.dice_weight * dice_loss +
            self.neg_weight * neg_intersection_loss
        )

        metrics = {
            'loss_total': total_loss.item(),
            'intensity_loss': intensity_loss.item(),
            'dice_loss': dice_loss.item(),
            'dice_score': bin_dice.item(),
            'soft_dice': soft_dice.item(),
            'neg_loss': neg_intersection_loss.item()
        }
        return total_loss, metrics


class OrientationLoss(nn.Module):
    """
    Specialized Loss for OrientationUNet3D:
    Trains the network to predict continuous 3D unit tangent vectors O(x) = (vz, vy, vx).
    - Symmetric Cosine Alignment Loss: 1 - |u . v|
    - Valid fiber foreground masking with contact-floor weighting
    """
    def __init__(self, baseline_floor=0.10, eps=1e-8):
        super().__init__()
        self.baseline_floor = baseline_floor
        self.eps = eps

    def forward(self, pred_dir, gt_dir, gt_intensity):
        """
        pred_dir:     (B, 3, D, H, W) unit vectors
        gt_dir:       (B, 3, D, H, W) unit vectors
        gt_intensity: (B, 1, D, H, W) potential field magnitude
        """
        dot = torch.abs(torch.sum(pred_dir * gt_dir, dim=1, keepdim=True))
        dot_clamped = torch.clamp(dot, 0.0, 1.0)

        # Valid foreground mask
        gt_ori_norm = torch.linalg.norm(gt_dir, dim=1, keepdim=True)
        valid_ori_mask = (gt_ori_norm > 1e-4).float()

        # Weight by magnitude |gt_intensity| with a baseline floor inside fiber volume
        weights = torch.clamp(torch.abs(gt_intensity) + self.baseline_floor * valid_ori_mask, 0.0, 1.0) * valid_ori_mask
        weight_sum = weights.sum()

        if weight_sum > 0:
            ori_loss = (weights * (1.0 - dot_clamped)).sum() / weight_sum
            mean_dot = (weights * dot_clamped).sum() / weight_sum
            mean_angle_rad = torch.acos(torch.clamp(mean_dot, 0.0, 1.0))
            mean_angle_deg = mean_angle_rad * (180.0 / torch.pi)
        else:
            ori_loss = torch.tensor(0.0, device=pred_dir.device)
            mean_angle_deg = torch.tensor(0.0, device=pred_dir.device)

        metrics = {
            'loss_total': ori_loss.item(),
            'ori_loss': ori_loss.item(),
            'mean_angle_deg': mean_angle_deg.item()
        }
        return ori_loss, metrics
