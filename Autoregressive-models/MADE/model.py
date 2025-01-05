import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8)))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)

class MADE(nn.Module):
    def __init__(self, in_dim, hidden_sizes, out_dim, num_masks=1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_sizes = hidden_sizes
        self.out_dim = out_dim
        self.num_masks = num_masks
        self.seed = 0

        self.net = nn.Sequential(
            MaskedLinear(in_dim, hidden_sizes[0]),
            nn.ReLU(),
            MaskedLinear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            MaskedLinear(hidden_sizes[1], out_dim),
            nn.Sigmoid()
        )

        self.m = {}
        self.update_masks()

    def update_masks(self):
        self.m[-1] = np.arange(1, self.in_dim + 1)

        for l in range(len(self.hidden_sizes)):
            self.m[l] = np.random.randint(1, self.m[l-1].max() + 1,
                                        size=self.hidden_sizes[l])

        self.m[len(self.hidden_sizes)] = np.arange(1, self.out_dim + 1)

        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]

        for l in range(len(layers)):
            if l == 0:
                mask = self.m[l].reshape(-1, 1) >= self.m[-1]
            elif l == len(layers) - 1:
                mask = self.m[len(self.hidden_sizes)].reshape(-1, 1) > self.m[l-1]
            else:
                mask = self.m[l].reshape(-1, 1) >= self.m[l-1]

            layers[l].set_mask(mask)

    def forward(self, x):
        return self.net(x)

    def next_mask(self):
        self.seed = (self.seed + 1) % self.num_masks
        np.random.seed(self.seed)
        self.update_masks()