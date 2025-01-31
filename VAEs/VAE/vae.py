import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from tqdm import tqdm

@dataclass
class Config:
    input_dim: int = 784
    hidden_dims = [128, 32, 16, 4]
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    act_fun: type = nn.GELU
    batch_size: int = 256
    epochs: int = 50
    lr: float = 1e-3


tensor_transform = transforms.ToTensor()


MNIST_dataset = datasets.MNIST(root = "./data",
									train = True,
									download = True,
									transform = tensor_transform)

MNIST_loader = torch.utils.data.DataLoader(dataset = MNIST_dataset,
							   batch_size = Config.batch_size,
								 shuffle = True)


class VAE(nn.Module):
    def __init__(self, config: Config):
        super(VAE, self).__init__()
        self.config = config
        self.encoder = nn.Sequential()
        layers = []
        prev_dim = config.input_dim
        self.z_size = self.config.hidden_dims[-1] // 2

        for dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(config.act_fun())
            prev_dim = dim
        
        layers = layers[:-1]
        self.encoder = nn.Sequential(*layers)

        self.decoder = nn.Sequential()
        layers = []
        prev_dim = self.z_size

        for dim in reversed(config.hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(config.act_fun())
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, config.input_dim))
        layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*layers)


    
    def reparameterize(self, mean, logvar, n_samples_per_z=1):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std.unsqueeze(1).repeat(1, n_samples_per_z, 1).view(-1, std.shape[-1]))
        z = mean.unsqueeze(1).repeat(1, n_samples_per_z, 1).view(-1, mean.shape[-1]) + eps * std.unsqueeze(1).repeat(1, n_samples_per_z, 1).view(-1, std.shape[-1])
        return z  


    def encode(self, x):
      mean, logvar = torch.split(self.encoder(x), split_size_or_sections=[self.z_size, self.z_size], dim=-1)
      return mean, logvar

    def decode(self, z):
        probs = self.decoder(z)
        return probs

    def forward(self, x, n_samples_per_z=1):
        mean, logvar = self.encode(x)

        batch_size, latent_dim = mean.shape
        if n_samples_per_z > 1:
            mean = mean.unsqueeze(1).expand(batch_size, n_samples_per_z, latent_dim)
            logvar = logvar.unsqueeze(1).expand(batch_size, n_samples_per_z, latent_dim)

            mean = mean.contiguous().view(batch_size * n_samples_per_z, latent_dim)
            logvar = logvar.contiguous().view(batch_size * n_samples_per_z, latent_dim)

        z = self.reparameterize(mean, logvar, n_samples_per_z)
        x_probs = self.decode(z)


        x_probs = x_probs.reshape(batch_size, n_samples_per_z, -1)


        x_probs = torch.mean(x_probs, dim=1)

        return {
            "imgs": x_probs,
            "z": z,
            "mean": mean,
            "logvar": logvar
        }
# ### Test
# hidden_dims = [128, 64, 36, 18, 18]
# input_dim = 256
# test_tensor = torch.randn([1, input_dim]).to(Config.device)

# config = Config(input_dim=input_dim, hidden_dims=hidden_dims)

# vae_test = VAE(config=config).to(Config.device)

# with torch.no_grad():
#   test_out = vae_test(test_tensor) 
#   print(test_out["imgs"].shape)   
        


# vae = VAE(config=Config(input_dim=18, hidden_dims=[18, 18])).to(Config.device)
# mean = torch.randn([1, 18]).to(Config.device)
# logvar = torch.randn([1, 18]).to(Config.device)
# # print(vae.reparameterize(mean, logvar, n_samples_per_z=1))

# print(torch.rand_like(logvar.unsqueeze(1).repeat(1,1,1).view(-1, logvar.shape[-1])).shape)

# vae = VAE(config=Config(input_dim=256, hidden_dims=[128, 64, 32, 16])).to(Config.device)
# mean = torch.randn([1, 256]).to(Config.device)
# logvar = torch.randn([1, 256]).to(Config.device)
# print(vae.forward(mean, n_samples_per_z=1)["imgs"].shape)

class SGVB(nn.Module):
    def __init__(self, device):
        super(SGVB, self).__init__()
        self.device = device
        self.log2pi = torch.log(2.0 * torch.tensor(np.pi)).to(device)
        self.torch_zero = torch.tensor(0.0).to(device)

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        # log P(x|μ,σ) = log((1/√(2πσ²)) * exp(-(x-μ)²/(2σ²)))
        # log((1/√(2πσ²)) = -1/2 * log(2πσ²)
        # log(exp(-(x-μ)²/(2σ²))) = -1/2 * (log(2π) + log(σ²)) - (x-μ)²/(2σ²)

        squared_diff = torch.square(sample - mean)
        var = torch.exp(logvar)
        log_prob = self.log2pi + logvar + (squared_diff / var)
        return -0.5 * torch.sum(log_prob, dim=raxis)


    def loss_SGVB(self, output):
        logpz = self.log_normal_pdf(output['z'], self.torch_zero, self.torch_zero)
        logqz_x = self.log_normal_pdf(output['z'], output['mean'], output['logvar'])
        return logpz -logqz_x

class KL_WO_E(nn.Module):
    def __init__(self, device):
        super(KL_WO_E, self).__init__()
        self.device = device

    def loss_KL_WO_E(self, output):
        logvar = output['logvar']
        mean = output['mean']
        var = torch.exp(logvar)
        return 0.5 * torch.sum(var + mean ** 2 - 1.0 - logvar, dim=1)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.sgvb = SGVB(config.device)
        self.kl_wo_e = KL_WO_E(config.device)
        self.model = VAE(config).to(config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.mse = nn.MSELoss(reduction='mean')
    
    def loss_func(self, x, reg_type='sgvb'):
        output = self.model(x)
        reconstruction_loss = self.mse(output['imgs'], x)
        reconstruction_loss = -1.0 * torch.sum(reconstruction_loss)
        
        # Regularization term
        if reg_type == 'sgvb':
            reg_loss = self.sgvb.loss_SGVB(output)
        elif reg_type == 'kl':
            reg_loss = self.kl_wo_e.loss_KL_WO_E(output)
        else:
            reg_loss = torch.tensor(0.0).to(self.device)

        # Total loss is reconstruction loss + regularization term
        coeff = 1e-3
        return -1.0 * torch.mean(reconstruction_loss + coeff * reg_loss)


    def train(self, dataloader, reg_type='sgvb'):
        self.model.train()
        losses = []
        
        for epoch in tqdm(range(self.config.epochs), desc='Epochs'):
            running_loss = 0.0
            batch_progress = tqdm(dataloader, desc='Batches', leave=False)
            
            for iter, (images, labels) in enumerate(batch_progress):
                batch_size = images.shape[0]
                images = images.reshape(batch_size, -1).to(self.device)
                loss = self.loss_func(images)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                avg_loss = running_loss / len(MNIST_dataset) * batch_size
                losses.append(loss.item())

            tqdm.write(f'Epoch [{epoch+1}/{self.config.epochs}], Average Loss: {avg_loss:.4f}')
        
        return losses

    def train_with_sgvb(self, dataloader):
        print("Training with SGVB regularization...")
        losses = self.train(dataloader, reg_type='sgvb')
        print("\nVisualizing SGVB results...")
        evaluate_model(self.model)
        plot_latent_images(self.model, n=20)
        return losses
    
    def train_with_kl_wo_e(self, dataloader):
        print("Training with KL (without expectation) regularization...")
        self.model = VAE(self.config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        losses = self.train(dataloader, reg_type='kl')
        print("\nVisualizing KL results...")
        evaluate_model(self.model)
        plot_latent_images(self.model, n=20)
        return losses

def plot_latent_images(model, n, digit_size=28):
    grid_x = np.linspace(-2, 2, n)
    grid_y = np.linspace(-2, 2, n)

    image_width = digit_size * n
    image_height = digit_size * n
    image = np.zeros((image_height, image_width))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = torch.tensor([[xi, yi]], dtype=torch.float32).to(model.config.device)
            with torch.no_grad():
                x_decoded = model.decode(z)
            digit = x_decoded.view(digit_size, digit_size).cpu().numpy()
            image[i * digit_size: (i + 1) * digit_size,
                  j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='Greys_r')
    plt.axis('Off')
    plt.title('Latent Space Visualization')
    plt.show()

def evaluate_model(model):
    original_imgs = torch.cat([MNIST_dataset[i][0] for i in range(5)])
    with torch.no_grad():
        res = model(original_imgs.reshape(5, -1).to(model.config.device))
        reconstructed_imgs = res['imgs']
        reconstructed_imgs = reconstructed_imgs.cpu().reshape(*original_imgs.shape)

    fig, axes = plt.subplots(5, 2, figsize=(10, 25))

    for i in range(5):
        original_image = original_imgs[i].reshape(28, 28)
        axes[i, 0].imshow(original_image, cmap='gray')
        axes[i, 0].set_title(f'Original Image {i+1}')
        axes[i, 0].axis('off')

        reconstructed_image = reconstructed_imgs[i].reshape(28, 28)
        axes[i, 1].imshow(reconstructed_image, cmap='gray')
        axes[i, 1].set_title(f'Reconstructed Image {i+1}')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

config = Config()
trainer = Trainer(config)
trainer.train_with_sgvb(MNIST_loader)
trainer.train_with_kl_wo_e(MNIST_loader)
