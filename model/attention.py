import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ChannelAttention(nn.Module):
    """
        Channel Attention
        Reference:
            Dual Attention Network for Scene Segmentation, Jun Fu et al. in CVPR2019
            arXiv: https://arxiv.org/pdf/1809.02983.pdf
    """

    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps. shape => (N, C, T, H, W)
            returns :
                out : attention value + input feature
                attention: (N, C, C) 
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class TemporalAttention(nn.Module):
    """
        Channel Attention
        Reference:
            Dual Attention Network for Scene Segmentation, Jun Fu et al. in CVPR2019
            arXiv: https://arxiv.org/pdf/1809.02983.pdf
    """

    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps. shape => (N, C, T, H, W)
            returns :
                out : attention value + input feature
                attention: (N, C, C) 
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out
