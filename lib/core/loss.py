import torch
from torch import nn
import torch.nn.functional as F

def soft_skeletonize(x, thresh_width=5):
    '''
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - maximal expected width of vessel
    '''
    for i in range(thresh_width):
        p1 = torch.nn.functional.max_pool3d(x * -1, (3, 1, 1), 1, (1, 0, 0)) * -1
        p2 = torch.nn.functional.max_pool3d(x * -1, (1, 3, 1), 1, (0, 1, 0)) * -1
        p3 = torch.nn.functional.max_pool3d(x * -1, (1, 1, 3), 1, (0, 0, 1)) * -1
        min_pool_x = torch.min(torch.min(p1, p2), p3)
        contour = torch.nn.functional.relu(torch.nn.functional.max_pool3d(min_pool_x, (3, 3, 3), 1, 1) - min_pool_x)
        x = torch.nn.functional.relu(x - contour)
    return x

def positive_intersection(center_line, vessel):
    '''
    inputs shape  (batch, channel, height, width)
    intersection formalized by first ares
    x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
    '''
    clf = center_line.view(*center_line.shape[:2], -1)
    vf = vessel.view(*vessel.shape[:2], -1)

    intersection = (clf * vf).sum(-1)
    return (intersection.sum(0) + 1e-12) / (clf.sum(-1).sum(0) + 1e-12)

def soft_cldice(pred, target):
    '''
    inputs shape  (batch, channel, height, width).
    calculate clDice acc
    '''
    target_skeleton = soft_skeletonize(target)
    cl_pred = soft_skeletonize(pred)
    clrecall = positive_intersection(target_skeleton, pred)  # ClRecall
    recall = positive_intersection(target, pred)
    clacc = positive_intersection(cl_pred, target)
    acc = positive_intersection(pred, target)
    return clrecall[0], clacc[0], recall[0], acc[0]

class ClDiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, pred, target):
        clrecall, clacc, recall, acc = soft_cldice(pred, target)
        cldice = (2. * clrecall * clacc) / (clrecall + clacc)
        return 1-cldice

class DiceLoss(torch.nn.Module):
    def __init__(self, eps=1.0) -> None:
        super().__init__()
        self.eps = eps

    def dice_coefficient(self, y_pred, y_true):
        self.eps = 1.0
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return dice

    def forward(self, y_pred, y_true):
        return 1 - self.dice_coefficient(y_pred, y_true)

def get_loss(args):
    return ClDiceLoss()

if __name__ == '__main__':
    loss_fn = ClDiceLoss()
    mask = torch.rand((1,1,64,64,64))
    pred_mask = torch.rand_like(mask)
    loss = loss_fn(mask, pred_mask)
    print(loss.item())