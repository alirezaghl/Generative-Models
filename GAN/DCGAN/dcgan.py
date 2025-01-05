import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="DCGAN Training")
    parser.add_argument('--lrG', type=float, default=2e-4, help='Learning rate of Generator')
    parser.add_argument('--lrD', type=float, default=4e-4, help='Learning rate of Discrimniator')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--image_size', type=int, default=64, help='Image size')
    parser.add_argument('--output_channel', type=int, default=1, help='Output channels for the generator (1 for grayscale)')
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimensionality of the latent space')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--features', type=int, default=64, help='Number of feature maps')
    return parser.parse_args()

# Generator Model
class Generator(nn.Module):
    def __init__(self, latent_dim, features, in_channel):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, features * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(features * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(features * 16, features * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(features * 8, features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(features * 4, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(features * 2, in_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, in_channel, features):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=2, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize((args.image_size, args.image_size)),
    ])
    
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    # Initialize Generator and Discriminator
    generator = Generator(args.latent_dim, args.features, args.output_channel).to(device)
    discriminator = Discriminator(args.output_channel, args.features).to(device)

    # Initialize Optimizers
    opt_gen = optim.Adam(generator.parameters(), lr=args.lrG, betas=(0.5, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=args.lrD, betas=(0.5, 0.999))
    
    criterion = nn.BCELoss()

    # Lists to store losses
    generator_losses = []
    discriminator_losses = []

    # Training Loop
    for epoch in range(args.epochs):
        for batch_idx, (data, _) in enumerate(trainloader):
            batch_size = data.size(0)
            data = data.to(device)
            
            ### Train Discriminator
            noise = torch.randn(batch_size, args.latent_dim, 1, 1).to(device)
            fake = generator(noise)
            
            disc_real = discriminator(data).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = discriminator(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            
            discriminator.zero_grad()
            loss_disc.backward()
            opt_disc.step()
            
            ### Train Generator
            output = discriminator(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            
            generator.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            
            # Save the losses
            generator_losses.append(loss_gen.item())
            discriminator_losses.append(loss_disc.item())
            
            # Print losses occasionally
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch {batch_idx}/{len(trainloader)} \
                      Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")
    
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
    print('Model Saved!')

    # Plot the losses
    plt.figure(figsize=(10, 5))
    plt.plot(generator_losses, label="Generator Loss")
    plt.plot(discriminator_losses, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Generator and Discriminator Loss During Training")
    plt.show()

if __name__ == "__main__":
    main()