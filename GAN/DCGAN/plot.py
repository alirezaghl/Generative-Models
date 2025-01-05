import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from dcgan import Generator

# Load the generator model
latent_dim = 100
features = 64
output_channel = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(latent_dim, features, output_channel).to(device)
generator.load_state_dict(torch.load('generator.pth', map_location=device))
generator.eval()

# Visualize the generated images
def visualize_samples(generator, num_samples=16, latent_dim=100):
    noise = torch.randn(num_samples, latent_dim, 1, 1).to(device)
    with torch.no_grad():
        fake_images = generator(noise).detach().cpu()
    
    plt.figure(figsize=(8, 8))
    for i in range(num_samples):
        plt.subplot(4, 4, i + 1)
        plt.imshow(fake_images[i].squeeze(0), cmap='gray')
        plt.axis('off')
    plt.show()

# Interpolation between two random latent vectors
def interpolate(generator, latent_dim=100, steps=10):
    z1 = torch.randn(1, latent_dim, 1, 1).to(device)
    z2 = torch.randn(1, latent_dim, 1, 1).to(device)
    z_interp = [z1 + (z2 - z1) * alpha for alpha in np.linspace(0, 1, steps)]

    interpolated_images = []
    for z in z_interp:
        with torch.no_grad():
            fake_image = generator(z).detach().cpu()
            interpolated_images.append(fake_image.squeeze(0))

    plt.figure(figsize=(2 * steps, 2))
    for i, img in enumerate(interpolated_images):
        plt.subplot(1, steps, i + 1)
        plt.imshow(img.squeeze(0), cmap='gray')
        plt.axis('off')
    plt.show()

# Visualize some generated samples
visualize_samples(generator, num_samples=16, latent_dim=latent_dim)

# Interpolate between two random latent vectors
interpolate(generator, latent_dim=latent_dim, steps=10)