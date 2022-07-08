import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence


class AreaBinsChamferLoss(nn.Module):  # the Chamfer distance as loss function for
    def __init__(self):                # the training of the prediction of the area-bin-centers
        super().__init__()
        self.name = "Chamfer-distance"

    def forward(self, areabins, gt_vsa_set):
        areabin_centers = 0.5 * (areabins[:, 1:] + areabins[:, :-1])
        n, p = areabin_centers.shape
        input_points = areabin_centers.view(n, p, 1)

        target_points = gt_vsa_set 
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(gt_vsa_set.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss
