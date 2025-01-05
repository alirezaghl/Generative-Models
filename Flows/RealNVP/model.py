import torch
import torch.nn as nn

class CouplingLayer(nn.Module):
    def __init__(self, mask, input_dim, hidden_dim):
        super(CouplingLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.register_buffer('mask', mask)

        self.s_fc1 = nn.Linear(input_dim, hidden_dim)
        self.s_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.s_fc3 = nn.Linear(hidden_dim, input_dim)

        self.t_fc1 = nn.Linear(input_dim, hidden_dim)
        self.t_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.t_fc3 = nn.Linear(hidden_dim, input_dim)

    def compute_s(self, x):
        s = torch.relu(self.s_fc1(x * self.mask))
        s = torch.relu(self.s_fc2(s))
        s = self.s_fc3(s)
        return s

    def compute_t(self, x):
        t = torch.relu(self.t_fc1(x * self.mask))
        t = torch.relu(self.t_fc2(t))
        t = self.t_fc3(t)
        return t

    def forward(self, x):
        s = self.compute_s(x)
        t = self.compute_t(x)
        z = x * self.mask + (1 - self.mask) * (x * torch.exp(s) + t)
        logdet = torch.sum((1 - self.mask) * s, dim=1)
        return z, logdet

    def reverse(self, z):
        s = self.compute_s(z)
        t = self.compute_t(z)
        x = z * self.mask + (1 - self.mask) * ((z - t) * torch.exp(-s))
        inv_logdet = torch.sum((1 - self.mask) * (-s), dim=1)
        return x, inv_logdet

def create_checkerboard_mask_2d(input_dim, invert=False):
    mask = torch.arange(input_dim) % 2
    if invert:
        mask = 1 - mask
    return mask.to(torch.float32)

class RealNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(RealNVP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.masks = [create_checkerboard_mask_2d(input_dim, invert=(i % 2 == 0)) for i in range(num_layers)]
        self.coupling_layers = nn.ModuleList(
            [CouplingLayer(self.masks[i], input_dim, hidden_dim) for i in range(num_layers)]
        )
        self.base_dist = torch.distributions.MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))

    def forward(self, x):
        logdet = 0
        for layer in self.coupling_layers:
            x, layer_logdet = layer(x)
            logdet += layer_logdet
        return x, logdet

    def reverse(self, z):
        logdet = 0
        for layer in reversed(self.coupling_layers):
            z, layer_logdet = layer.reverse(z)
            logdet += layer_logdet
        return z, logdet

    def sample(self, batch_size):
        z = self.base_dist.sample((batch_size,))
        x, _ = self.reverse(z)
        return x

    def log_prob(self, x):
        z, logdet = self.forward(x)
        log_prob = self.base_dist.log_prob(z) + logdet
        return log_prob
