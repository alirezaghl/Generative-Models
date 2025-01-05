import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from model import RealNVP
from parser import parse_args


model = RealNVP(input_dim=2, hidden_dim=256, num_layers=6)
model.load_state_dict(torch.load('realnvp.pth'))
model.eval()

args = parse_args()
x, y = make_moons(n_samples=args.n_samples, noise=args.noise)
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
dataset = TensorDataset(x, y)
train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)



with torch.no_grad():
    samples = model.sample(1000).numpy()
    
plt.figure(figsize=(8, 8))
plt.scatter(x[:, 0], x[:, 1], c='blue', label='Real Data', alpha=0.5)
plt.scatter(samples[:, 0], samples[:, 1], c='red', label='Generated Data', alpha=0.5)
plt.legend()
plt.title('Real vs Generated Data (Moons)')
plt.show()


real_data = next(iter(train_loader))[0]
with torch.no_grad():
    z_real, _ = model.forward(real_data)


z_sample = torch.randn((500, 2))  
with torch.no_grad():
    generated_data, _ = model.reverse(z_sample)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].scatter(real_data[:, 0], real_data[:, 1], c='red')
axes[0, 0].set_title("Inference data space X")
axes[0, 0].set_xlim([-2, 2])
axes[0, 0].set_ylim([-1.5, 1.5])

axes[0, 1].scatter(z_real[:, 0], z_real[:, 1], c='red')
axes[0, 1].set_title("Inference latent space Z")
axes[0, 1].set_xlim([-4, 4])
axes[0, 1].set_ylim([-4, 4])

axes[1, 0].scatter(z_sample[:, 0], z_sample[:, 1], c='green')
axes[1, 0].set_title("Generated latent space Z")
axes[1, 0].set_xlim([-4, 4])
axes[1, 0].set_ylim([-4, 4])

axes[1, 1].scatter(generated_data[:, 0], generated_data[:, 1], c='green')
axes[1, 1].set_title("Generated data space X")
axes[1, 1].set_xlim([-2, 2])
axes[1, 1].set_ylim([-1.5, 1.5])

plt.tight_layout()
plt.show()