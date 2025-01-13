import torch
import matplotlib.pyplot as plt
import numpy as np

def generate_new_samples(model, num_samples=64, H=28, W=28):
   
    model.eval()
    
    samples = torch.zeros(size=(num_samples, 1, H, W)).to(next(model.parameters()).device)
    
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                # Skip first row and column (they stay zero)
                if j > 0 and i > 0:
                    # Get model predictions
                    out = model(samples)
                    # Sample from predicted probabilities
                    samples[:, :, i, j] = torch.bernoulli(
                        out[:, :, i, j], 
                        out=samples[:, :, i, j]
                    )
    
    return samples.cpu().numpy().transpose(0, 2, 3, 1)

def plot_sample_grid(samples, grid_size=8, figsize=(15, 15)):
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    
    for i in range(grid_size * grid_size):
        sample = samples[i]
        row, col = divmod(i, grid_size)
        axes[row, col].imshow(sample, cmap='gray')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def generate_and_visualize(model, num_samples=64):

    samples = generate_new_samples(model, num_samples)
    
    plot_sample_grid(samples)
    
    return samples

if __name__ == "__main__":
    from config import device
    from pixel_cnn import PixelCNN
    
    model = PixelCNN().to(device)
    model.load_state_dict(torch.load('pixelcnn_model.pth'))
    
    samples = generate_and_visualize(model)