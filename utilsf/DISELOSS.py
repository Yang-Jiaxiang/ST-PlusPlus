import torch
import torch.nn.functional as F

def dice_coefficient(pred, target, epsilon=1e-6):
    pred = torch.argmax(pred, dim=1)  # shape: [128, 400, 400]

    # Flatten the tensors
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    dice_loss = 1 - dice
    return dice_loss.requires_grad_(True)
