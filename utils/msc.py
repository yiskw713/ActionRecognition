import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialMSC(nn.Module):
    """
    Spatial multi-scale inputs
    """

    def __init__(self, model, scales=[1.0, 0.75, 0.5], val_msc=False):
        super().__init__()
        self.model = model
        self.scales = scales
        self.val_msc = val_msc  # if you want to input multi-scale videos when validation

    def forward(self, x):
        if self.training or self.val_msc:
            # Scaled
            scaled_logits = []
            for s in self.scales:
                if s == 1.0:
                    xx = x
                else:
                    xx = F.interpolate(
                        x, scale_factor=s, mode="bilinear", align_corners=False)
                scaled_logits.append(self.model(xx))

            # Max pooling
            max_logits = torch.max(torch.stack(scaled_logits), dim=0)[0]

            if self.training:
                return scaled_logits + [max_logits]
            else:
                return max_logits

        else:
            return self.model(x)


class TemporalMSC(nn.Module):
    """
    Temporal multi-scale inputs. (eg. 64f, 32f, 16f)
    """

    def __init__(self, model, scales=[1.0, 0.5, 0.25], val_msc=False):
        super().__init__()
        self.model = model
        self.scales = scales
        self.val_msc = val_msc  # if you want to input multi-scale videos when validation

    def forward(self, x):
        if self.training or self.val_msc:
            _, _, T, _, _ = x.shape

            # Scaled
            scaled_logits = []
            for s in self.scales:
                if s == 1.0:
                    xx = x
                else:
                    start_frame = np.random.randint(0, T - int(s * T) + 1)
                    xx = x[:, :, start_frame:start_frame + int(s * T)]
                    scaled_logits.append(self.model(xx))

            # Max pooling
            max_logits = torch.max(torch.stack(scaled_logits), dim=0)[0]

            if self.training:
                return scaled_logits + [max_logits]
            else:
                return max_logits

        else:
            return self.model(x)


class SpatioTemporalMSC(nn.Module):
    """
    Spatiotemporal multi-scale inputs
    """

    def __init__(self, model, spatial_scales=[1.0, 0.75], temporal_scales=[1.0, 0.5], val_msc=False):
        super().__init__()
        self.model = model
        self.spatial_scales = spatial_scales
        self.temporal_scales = temporal_scales
        self.val_msc = val_msc  # if you want to input multi-scale videos when validation

    def forward(self, x):
        if self.training or self.val_msc:
            _, _, T, _, _ = x.shape

            # Scaled
            scaled_logits = []
            for s in self.spatial_scales:
                if s == 1.0:
                    sx = x
                else:
                    sx = F.interpolate(
                        x, scale_factor=s, mode="bilinear", align_corners=False)

                for t in self.temporal_scales:
                    if s == 1.0:
                        stx = sx
                    else:
                        start_frame = np.random.randint(0, T - int(t * T) + 1)
                        stx = sx[:, :, start_frame:start_frame + int(t * T)]

                    scaled_logits.append(self.model(stx))

            # Max pooling
            max_logits = torch.max(torch.stack(scaled_logits), dim=0)[0]

            if self.training:
                return scaled_logits + [max_logits]
            else:
                return max_logits

        else:
            return self.model(x)
