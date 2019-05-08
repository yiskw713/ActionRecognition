import numpy as np
import torch
import torch.nn as nn


class MSC(nn.Module):
    """
    Temporal multi-scale inputs. (eg. 64f, 32f, 16f)
    """

    def __init__(self, model, scales=[0.5, 0.25]):
        super(MSC, self).__init__()
        self.model = model
        self.scales = scales

    def forward(self, x):
        # Original scale
        _, _, T, _, _ = x.shape
        logit = self.model(x)

        # Scaled
        scaled_logits = []
        for s in self.scales:
            start_frame = np.random.randint(1, T - int(s * T) + 1)
            xx = x[:, :, start_frame:start_frame + int(s * T)]
            scaled_logits.append(self.model(xx))

        # Pixel-wise max
        all_logits = [logit] + scaled_logits
        max_logits = torch.max(torch.stack(all_logits), dim=0)[0]

        if self.training:
            return all_logits + [max_logits]
        else:
            return max_logits
