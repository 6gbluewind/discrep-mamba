import torch
import numpy as np
import torch.nn as nn

class SeparableWaveletTransform3D(nn.Module):
    def __init__(self, in_channels, device):
        super().__init__()
        self.in_channels = in_channels
        self.device = device

        # Create a 1D filter (Haar)
        sqrt2 = np.sqrt(2)
        harr_wav_L = torch.tensor([1.0 / sqrt2, 1.0 / sqrt2], dtype=torch.float32, device=device)
        harr_wav_H = torch.tensor([-1.0 / sqrt2, 1.0 / sqrt2], dtype=torch.float32, device=device)

        # Depth-dimensional convolution
        self.conv_d_L = nn.Conv3d(in_channels, in_channels, kernel_size=(2, 1, 1),
                                  stride=(2, 1, 1), padding=0, bias=False, groups=in_channels)
        self.conv_d_H = nn.Conv3d(in_channels, in_channels, kernel_size=(2, 1, 1),
                                  stride=(2, 1, 1), padding=0, bias=False, groups=in_channels)

        # Height-dimensional convolution
        self.conv_h_L = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 2, 1),
                                  stride=(1, 2, 1), padding=0, bias=False, groups=in_channels)
        self.conv_h_H = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 2, 1),
                                  stride=(1, 2, 1), padding=0, bias=False, groups=in_channels)

        # Width-dimension convolution
        self.conv_w_L = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 2),
                                  stride=(1, 1, 2), padding=0, bias=False, groups=in_channels)
        self.conv_w_H = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 2),
                                  stride=(1, 1, 2), padding=0, bias=False, groups=in_channels)

        self._init_weights(harr_wav_L, harr_wav_H)
        self.to(device)

    def _init_weights(self, wav_L, wav_H):
        with torch.no_grad():
            # Depth
            self.conv_d_L.weight.data = wav_L.view(1, 1, 2, 1, 1).expand(self.in_channels, -1, -1, -1, -1)
            self.conv_d_H.weight.data = wav_H.view(1, 1, 2, 1, 1).expand(self.in_channels, -1, -1, -1, -1)
            # Height
            self.conv_h_L.weight.data = wav_L.view(1, 1, 1, 2, 1).expand(self.in_channels, -1, -1, -1, -1)
            self.conv_h_H.weight.data = wav_H.view(1, 1, 1, 2, 1).expand(self.in_channels, -1, -1, -1, -1)
            # Width
            self.conv_w_L.weight.data = wav_L.view(1, 1, 1, 1, 2).expand(self.in_channels, -1, -1, -1, -1)
            self.conv_w_H.weight.data = wav_H.view(1, 1, 1, 1, 2).expand(self.in_channels, -1, -1, -1, -1)

            # Freeze weights
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        if input.dtype != self.conv_d_L.weight.dtype:
            input = input.to(self.conv_d_L.weight.dtype)

        # Depth
        d_L = self.conv_d_L(input)
        d_H = self.conv_d_H(input)

        # Height
        d_L_h_L = self.conv_h_L(d_L)
        d_L_h_H = self.conv_h_H(d_L)
        d_H_h_L = self.conv_h_L(d_H)
        d_H_h_H = self.conv_h_H(d_H)

        # Width
        LLL = self.conv_w_L(d_L_h_L)
        LLH = self.conv_w_H(d_L_h_L)
        LHL = self.conv_w_L(d_L_h_H)
        LHH = self.conv_w_H(d_L_h_H)
        HLL = self.conv_w_L(d_H_h_L)
        HLH = self.conv_w_H(d_H_h_L)
        HHL = self.conv_w_L(d_H_h_H)
        HHH = self.conv_w_H(d_H_h_H)

        return LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH

