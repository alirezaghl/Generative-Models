import torch
import torch.nn as nn
import torch.nn.functional as F
from masked_conv import MaskedConv2d

class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_f, x_g = torch.chunk(x, 2, dim=1)
        return torch.tanh(x_f) * torch.sigmoid(x_g)

class PixelCNN(nn.Module):
    def __init__(self, input_channels=1, n_hidden=64, n_layers=8):
        super().__init__()

        # First layer with mask type 'A' to prevent seeing the current pixel
        self.first_conv = MaskedConv2d(
            in_c=input_channels,
            out_c=n_hidden,
            kernel_size=7,
            mask_type='A',
            padding=3
        )

        # Stack of hidden layers with mask type 'B'
        self.hidden_layers = nn.ModuleList([
            MaskedConv2d(
                in_c=n_hidden,
                out_c=2 * n_hidden,  # Doubling the output channels for gated activations
                kernel_size=7,
                mask_type='B',
                padding=3
            ) for _ in range(n_layers)
        ])

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(n_hidden) for _ in range(n_layers + 1)
        ])

        # Gated activation module
        self.gated_activation = GatedActivation()

        # Last convolution layer
        self.last_conv = MaskedConv2d(
            in_c=n_hidden,
            out_c=n_hidden,
            kernel_size=1,
            mask_type='B'
        )

        # Output convolution layer
        self.output_conv = MaskedConv2d(
            in_c=n_hidden,
            out_c=input_channels,
            kernel_size=1,
            mask_type='B'
        )

    def forward(self, x):
        # First layer
        x = self.first_conv(x)
        x = self.batch_norms[0](x)
        x = F.relu(x)

        for i, (conv, bn) in enumerate(zip(self.hidden_layers, self.batch_norms[1:])):
            residual = x
            x = conv(x)
            x = self.gated_activation(x)
            x = bn(x)
            if i > 0:  # Skip first layer for residual -> pixelcnn++
                x = x + residual

        # Last layer
        x = self.last_conv(x)
        x = self.batch_norms[-1](x)
        x = F.relu(x)

        # Output layer
        x = self.output_conv(x)
        return torch.sigmoid(x)