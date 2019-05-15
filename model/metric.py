import torch
import torch.nn as nn
import torch.nn.functional as F


class L2ConstrainedLinear(nn.Module):

    """
    L2-constrained layer for Softmax Loss.
    Reference:  L2-constrained Softmax Loss for Discriminative Face Verification,
                Rajeev Ranjan et al. CoRR, 2017
                arXiv: https://arxiv.org/pdf/1703.09507.pdf
    """

    def __init__(self, in_channel, n_classes, alpha=28):
        super().__init__()

        self.alpha = alpha
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_channel))

    def forward(self, x):
        """
        x: features from CNN. shape => (N, C)
        """

        x = F.normalize(x, p=2, dim=1)
        x = self.alpha * x
        x = F.linear(x, F.normalize(self.weight, p=2))

        return x
