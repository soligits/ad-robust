import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize

class ADScore(nn.Module):

    def __init__(self, mean_filter = torch.ones((1, 1, 3, 3)) / 3*3, scale_factors=[1, 2, 4, 8], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mean_filter = mean_filter
        self.scale_factors = scale_factors


    def forward(self, x1: torch.Tensor, x2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        multi_scale_err = torch.Tensor(x1.shape[1:])
        for k in self.scale_factors:
            s1 = resize(x1, x1.shape[1:] / k)
            s2 = resize(x2, x2.shape[1:] / k)
            err = torch.mean(torch.square(s1 - s2), dim=0)
            err = resize(err, x1.shape[1:])
            multi_scale_err += err
        multi_scale_err /= len(self.scale_factors)
        multi_scale_err = torch.squeeze(F.conv2d(torch.unsqueeze(multi_scale_err, dim=0), self.mean_filter, padding='same'), dim=0)
        return torch.max(torch.abs(multi_scale_err - t))
