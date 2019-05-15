import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : feature maps from feature extractor. (N, C, T, H, W)
        outputs :
            feature maps weighted by attention along a channel dimension
        """

        N, C, T, H, W = x.shape
        query = x.view(N, C, -1)    # (N, C, T*H*W)
        key = x.view(N, C, -1).permute(0, 2, 1)    # (N, T*H*W, C)

        # calculate correlation
        energy = torch.bmm(query, key)    # (N, C, C)
        energy = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy)

        value = x.view(N, C, -1)

        out = torch.bmm(attention, value)
        out = out.view(N, C, T, H, W)
        out = self.gamma * out + x
        return out


class TemporalAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : feature maps from feature extractor. (N, C, T, H, W)
        outputs :
            feature maps weighted by attention along a channel dimension
        """

        N, C, T, H, W = x.shape

        # (N, C, T*H*W)
        query = x.transpose(1, 2).contiguous().view(N, T, -1)
        # (N, T*H*W, C)
        key = x.transpose(1, 2).contiguous().view(N, T, -1).permute(0, 2, 1)

        # calculate correlation
        energy = torch.bmm(query, key)    # (N, T, T)
        energy = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy)

        value = x.transpose(1, 2).contiguous().view(N, T, -1)

        out = torch.bmm(attention, value)
        out = out.view(N, C, T, H, W)
        out = self.gamma * out + x
        return out
