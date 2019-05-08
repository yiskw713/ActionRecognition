import torch
import torch.nn as nn
import torch.nn.functional as F


class MSC(nn.Module):
    """
    Spatial multi-scale inputs
    """

    def __init__(self, model, scales=[0.75, 0.5]):
        super(MSC, self).__init__()
        self.model = model
        self.scales = scales

    def forward(self, x):
        # Original scale (1.)
        logit = self.model(x)

        # Scaled
        scaled_logits = []
        for s in self.scales:
            xx = F.interpolate(
                x, scale_factor=s, mode="bilinear", align_corners=False)
            scaled_logits.append(self.model(xx))

        # Pixel-wise max
        all_logits = [logit] + scaled_logits
        max_logits = torch.max(torch.stack(all_logits), dim=0)[0]

        if self.training:
            return all_logits + [max_logits]
        else:
            return max_logits
