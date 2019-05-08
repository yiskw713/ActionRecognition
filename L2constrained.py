import torch.nn as nn
import torch.nn.functionla as F


class L2ConstrainedLayer(nn.Module):

    """
    L2-constrained layer for Softmax Loss.
    Reference:  L2-constrained Softmax Loss for Discriminative Face Verification, 
                Rajeev Ranjan et al. CoRR, 2017
                arXiv: https://arxiv.org/pdf/1703.09507.pdf
    """

    def __init__(self, alpha=28):
        super().__init__()

        self.alpha = alpha

    def forward(self, x):
        """
        x: the output from fc layer. shape => (N, C) 
        """

        x = F.normalize(x, p=2, dim=1)
        x = self.alpha * x
        return x
