import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.same_channels = in_channels==out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        if self.same_channels:
            out = x + x2
        else:
            out = x1 + x2
        return out / 1.414

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class Unet(nn.Module):
    def __init__(self, in_channels, n_feat=256):
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(up1 + temb1, down2)
        up3 = self.up2(up2 + temb2, down1)
        
        out = self.out(torch.cat((up3, x), 1))
        return out

def ddpm_schedules(beta1, beta2, T):
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    alpha_t = 1 - beta_t
    alphabar_t = torch.cumprod(alpha_t, dim=0)
    sqrt_beta_t = torch.sqrt(beta_t)
    sqrtab = torch.sqrt(alphabar_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }

class DDPM(nn.Module):
    def __init__(self, nn_model, beta1, beta2, T, device, image_size=28):
        super().__init__()
        self.nn_model = nn_model
        self.beta1 = beta1
        self.beta2 = beta2
        self.T = T
        self.device = device
        self.image_size = image_size
        self.loss_mse = nn.MSELoss()

        self.scheduler = ddpm_schedules(beta1=beta1, beta2=beta2, T=T)
        self.register_buffer('alpha_t', self.scheduler['alpha_t'].to(device))
        self.register_buffer('oneover_sqrta', self.scheduler['oneover_sqrta'].to(device))
        self.register_buffer('sqrt_beta_t', self.scheduler['sqrt_beta_t'].to(device))
        self.register_buffer('alphabar_t', self.scheduler['alphabar_t'].to(device))
        self.register_buffer('sqrtab', self.scheduler['sqrtab'].to(device))
        self.register_buffer('sqrtmab', self.scheduler['sqrtmab'].to(device))
        self.register_buffer('mab_over_sqrtmab', self.scheduler['mab_over_sqrtmab'].to(device))

    def forward(self, x_0: torch.Tensor):
        x_0 = x_0.to(self.device)
        batch_size = x_0.shape[0]
        
        t = torch.randint(1, self.T + 1, (batch_size,), device=self.device)
        t = t.view(-1, 1, 1, 1)
        
        noise = torch.randn_like(x_0)
        x_t = self.sqrtab[t] * x_0 + self.sqrtmab[t] * noise  # q(xt|x0) equivalent for q(zt|zt-1)
        
        noise_pred = self.nn_model(x_t, t/self.T)
        
        return self.loss_mse(noise, noise_pred), x_t, noise

    @torch.no_grad()
    def backward(self, n_samples, n_channels):
        samples = torch.randn(n_samples, n_channels, self.image_size, self.image_size).to(self.device)
        intermediate = []
        
        for t in tqdm(reversed(range(1, self.T)), desc='Sampling'):
            t_tensor = (torch.ones(n_samples, device=self.device) * t).long()
            
            noise_pred = self.nn_model(samples, t_tensor / self.T)
            
            if t > 1:
                noise = torch.randn_like(samples)
            else:
                noise = 0
                
            samples = (
                (samples - self.mab_over_sqrtmab[t] * noise_pred) * self.oneover_sqrta[t]
                + self.sqrt_beta_t[t] * noise
            )
            
            if t % 100 == 0 or t == self.T-1:
                intermediate.append(samples.detach().cpu().clone())
                
        return samples, intermediate

def train_loop(ddpm, optimizer, train_loader, n_epochs, device, save_intermediate=True):
    for epoch in range(n_epochs):
        print(f'\nEpoch {epoch+1}/{n_epochs}')
        ddpm.train()
        
        # Linear learning rate decay
        optimizer.param_groups[0]['lr'] = lr * (1 - epoch/n_epochs)
        
        pbar = tqdm(train_loader, desc='Training')
        loss_ema = None
        
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)
            optimizer.zero_grad()
            
            loss, x_t, noise = ddpm(images)
            
            loss.backward()
            optimizer.step()
            
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
        
        if epoch % 5 == 0 or epoch == n_epochs-1:
            ddpm.eval()
            with torch.no_grad():
                n_sample = 20
                samples, intermediate = ddpm.backward(n_sample, n_channels=1)
                
                # Plot final samples
                fig, ax = plt.subplots(5, 4, figsize=(8, 10))
                for i, j in itertools.product(range(5), range(4)):
                    ax[i,j].get_xaxis().set_visible(False)
                    ax[i,j].get_yaxis().set_visible(False)
                    if i*4 + j < n_sample:
                        ax[i,j].imshow(samples[i*4 + j, 0].cpu(), cmap='gray')
                plt.suptitle(f'Epoch {epoch+1} Samples')
                plt.show()
                
                if save_intermediate and len(intermediate) > 0:
                    fig, ax = plt.subplots(4, 4, figsize=(8, 8))
                    steps_to_show = [0, len(intermediate)//3, 2*len(intermediate)//3, -1]
                    for i, step_idx in enumerate(steps_to_show):
                        for j in range(4):
                            ax[i,j].get_xaxis().set_visible(False)
                            ax[i,j].get_yaxis().set_visible(False)
                            ax[i,j].imshow(intermediate[step_idx][j, 0].cpu(), cmap='gray')
                            if j == 0:
                                ax[i,j].set_ylabel(f'Step {step_idx}')
                    plt.suptitle(f'Epoch {epoch+1} Intermediate Steps')
                    plt.show()
        
        if epoch == n_epochs-1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': ddpm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_ema,
            }, 'ddpm_mnist_final.pth')

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    n_epochs = 20
    batch_size = 128
    lr = 1e-4
    T = 400

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = Unet(in_channels=1, n_feat=128).to(device)
    ddpm = DDPM(nn_model=model, beta1=1e-4, beta2=0.02, T=T, device=device, image_size=28)
    ddpm.to(device)

    optimizer = torch.optim.Adam(ddpm.parameters(), lr=lr)

    train_loop(ddpm, optimizer, train_loader, n_epochs, device)