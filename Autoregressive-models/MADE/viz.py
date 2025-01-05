import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def plot_reconstructions_and_samples(model, test_dataset, device, num_samples=10):
    """Show both reconstructions and pure samples"""
    # Get random indices
    indices = torch.randperm(len(test_dataset))[:num_samples]
    
    # Get ground truth samples
    ground_truth = torch.stack([test_dataset[i][0] for i in indices]).to(device)
    
    model.eval()
    with torch.no_grad():
        # 1. Generate reconstructions
        reconstructed = ground_truth.clone()
        for i in range(model.in_dim):
            current = reconstructed.clone()
            current[:, i:] = 0  # mask out current and future pixels
            logits = model(current)
            probs = logits[:, i]
            reconstructed[:, i] = torch.bernoulli(probs)
        
        # 2. Generate pure samples
        pure_samples = torch.zeros(num_samples, model.in_dim).to(device)
        for i in range(model.in_dim):
            logits = model(pure_samples)
            probs = logits[:, i]
            pure_samples[:, i] = torch.bernoulli(probs)
    
    # Plot all three
    plt.figure(figsize=(20, 4))
    
    # Ground truth
    plt.subplot(1, 3, 1)
    grid_gt = make_grid(ground_truth.view(-1, 1, 28, 28).cpu(), nrow=5)
    plt.imshow(grid_gt.permute(1, 2, 0))
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Reconstructions
    plt.subplot(1, 3, 2)
    grid_recon = make_grid(reconstructed.view(-1, 1, 28, 28).cpu(), nrow=5)
    plt.imshow(grid_recon.permute(1, 2, 0))
    plt.title('Reconstructions')
    plt.axis('off')
    
    # Pure samples
    plt.subplot(1, 3, 3)
    grid_pure = make_grid(pure_samples.view(-1, 1, 28, 28).cpu(), nrow=5)
    plt.imshow(grid_pure.permute(1, 2, 0))
    plt.title('Pure Generated Samples')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_loss_curves(train_losses, test_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.grid(True)
    plt.show()