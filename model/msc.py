import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialMSC(nn.Module):
    """
    Spatial multi-scale inputs
    """

    def __init__(self, base, scales=[1.0, 0.75], val_msc=False):
        super().__init__()
        self.base = base
        self.scales = scales
        self.val_msc = val_msc  # if you want to input multi-scale videos when validation

    def forward(self, x):
        if self.training or self.val_msc:
            N, C, T, H, W = x.shape
            # Scaled
            scaled_logits = []
            for s in self.scales:
                if s == 1.0:
                    scaled_x = x
                else:
                    scaled_x = x.transpose(1, 2)    # (N, T, C, H, W)
                    scaled_x = scaled_x.contiguous().view(-1, C, H, W)
                    scaled_x = F.interpolate(
                        scaled_x, scale_factor=s, mode="bilinear", align_corners=False)
                    _, _, scaled_H, scaled_W = scaled_x.shape
                    scaled_x = scaled_x.view(N, T, C, scaled_H, scaled_W)
                    scaled_x = scaled_x.transpose(1, 2)  # (N, C, T, H, W)
                scaled_logits.append(self.base(scaled_x))

            # Max pooling
            max_logits = torch.max(torch.stack(scaled_logits), dim=0)[0]

            if self.training:
                return scaled_logits + [max_logits]
            else:
                return max_logits

        else:
            return self.base(x)


class TemporalMSC(nn.Module):
    """
    Temporal multi-scale inputs. (eg. 64f, 32f, 16f)
    """

    def __init__(self, base, scales=[1.0, 0.5], val_msc=False):
        super().__init__()
        self.base = base
        self.scales = scales
        self.val_msc = val_msc  # if you want to input multi-scale videos when validation

    def forward(self, x):
        if self.training or self.val_msc:
            _, _, T, _, _ = x.shape

            # Scaled
            scaled_logits = []
            for s in self.scales:
                if s == 1.0:
                    scaled_x = x
                    scaled_logits.append(self.base(scaled_x))
                else:
                    start_frame = np.random.randint(0, T - int(s * T) + 1)
                    scaled_x = x[:, :, start_frame:start_frame + int(s * T)]
                    scaled_logits.append(self.base(scaled_x))

            # Max pooling
            max_logits = torch.max(torch.stack(scaled_logits), dim=0)[0]

            if self.training:
                return scaled_logits + [max_logits]
            else:
                return max_logits

        else:
            return self.base(x)


class SpatioTemporalMSC(nn.Module):
    """
    Spatiotemporal multi-scale inputs
    """

    def __init__(self, base, spatial_scales=[1.0, 0.75], temporal_scales=[1.0, 0.5], val_msc=False):
        super().__init__()
        self.base = base
        self.spatial_scales = spatial_scales
        self.temporal_scales = temporal_scales
        self.val_msc = val_msc  # if you want to input multi-scale videos when validation

    def forward(self, x):
        if self.training or self.val_msc:
            N, C, T, H, W = x.shape

            # Scaled
            scaled_logits = []
            for s in self.spatial_scales:
                if s == 1.0:
                    sx = x
                else:
                    sx = x.transpose(1, 2)    # shape => (N, T, C, H, W)
                    sx = sx.contiguous().view(-1, C, H, W)
                    sx = F.interpolate(
                        sx, scale_factor=s, mode="bilinear", align_corners=False)
                    _, _, scaled_H, scaled_W = sx.shape
                    sx = sx.view(N, T, C, scaled_H, scaled_W)
                    sx = sx.transpose(1, 2)    # shape => (N, C, T, H, W)

                for t in self.temporal_scales:
                    if t == 1.0:
                        stx = sx
                    else:
                        start_frame = np.random.randint(0, T - int(t * T) + 1)
                        stx = sx[:, :, start_frame:start_frame + int(t * T)]

                    scaled_logits.append(self.base(stx))

            # Max pooling
            max_logits = torch.max(torch.stack(scaled_logits), dim=0)[0]

            if self.training:
                return scaled_logits + [max_logits]
            else:
                return max_logits

        else:
            return self.base(x)
