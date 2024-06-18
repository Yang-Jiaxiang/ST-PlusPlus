import numpy as np
from PIL import Image
import torch.nn as nn
import torch
import torch.nn.functional as F



def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, pred, targets, smooth=1): 
        target_one_hot = F.one_hot(targets, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        target_one_hot = target_one_hot[:, 1:, :, :]
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        pred = pred[:, 1:, :, :]  # shape: (16, C-1, 224, 224), e.g. (16, 3, 224, 224) -> (16, 2, 224, 224)
        pred = F.sigmoid(pred)
        
        #flatten label and prediction tensors
        pred_flat = pred.contiguous().view(-1)
        target_flat = target_one_hot.contiguous().view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice
    
class DiceCoefficient:
    def __init__(self):
        self.dice_scores = []

    def add_batch(self, predictions, gts):
        # 初始化 DiceLoss
        dice_loss_fn = DiceLoss()
        
        # 計算 Dice loss
        dice_loss = dice_loss_fn(predictions, gts)
        
        # 計算 Dice coefficient
        dice = 1 - dice_loss
        
        # 添加到列表中
        self.dice_scores.append(dice.item())

    def evaluate(self):
        return torch.mean(torch.tensor(self.dice_scores)).item()
    
class meanIOU:
    def __init__(self):
        self.miou_scores = []

    def add_batch(self, predictions, gts):
        # Apply threshold
        gts_one_hot = F.one_hot(gts, num_classes=predictions.shape[1]).permute(0, 3, 1, 2).float()
        gts_one_hot = gts_one_hot[:, 1:, :, :]
        
        predictions = predictions[:, 1:, :, :]
        predictions = F.sigmoid(predictions)    
        
        # Calculate intersection and union
        intersection = (predictions * gts).sum(dim=(2, 3))
        union = (predictions + gts).sum(dim=(2, 3)) - intersection
        
        # Calculate mIOU
        miou = (intersection + 1e-6) / (union + 1e-6)
        self.miou_scores.append(miou.mean().item())

    def evaluate(self):
        return torch.mean(torch.tensor(self.miou_scores)).item()
    
    
def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

    return cmap
