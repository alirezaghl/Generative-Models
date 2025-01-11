import torch
import matplotlib.pyplot as plt
import numpy as np

def generate_samples_from_batch(model, batch_images, H=28, W=28):
    
    model.eval()  
    
    with torch.no_grad():
        pred = model(batch_images)
        
        for i in range(H):
            for j in range(W):
                pred[:, :, i, j] = torch.bernoulli(pred[:, :, i, j], out=pred[:, :, i, j])
        
        samples = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    return samples

def plot_sample_grid(samples, grid_size=8, figsize=(15, 15)):

    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    
    for i in range(grid_size * grid_size):
        sample = samples[i]
        row, col = divmod(i, grid_size)
        axes[row, col].imshow(sample, cmap='gray')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_model_samples(model, test_loader, device):
   
    for images, _ in test_loader:
        images = images.to(device)
        samples = generate_samples_from_batch(model, images)
        # Plot the first 64 samples in an 8x8 grid
        plot_sample_grid(samples, grid_size=8)
        break  # Only process first batch

if __name__ == "__main__":
    
    from config import device, test_loader
    from pixel_cnn import PixelCNN
    
    model = PixelCNN().to(device)
    model.load_state_dict(torch.load('pixelcnn_model.pth'))
    
    visualize_model_samples(model, test_loader, device)