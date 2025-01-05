import torch
import matplotlib.pyplot as plt
from rbm import RBM, RBMConfig
from data import get_mnist_dataloader

def visualize_reconstructions(rbm, data, epoch):
    with torch.no_grad():
        recon_samples, recon_probs, _, _ = rbm.gibbs_sampling(data)
    
    fig, axes = plt.subplots(2, 8, figsize=(15, 4))
    
    for i in range(8):
        # Original images
        axes[0, i].imshow(data[i].cpu().view(28, 28), cmap='gray')
        axes[0, i].axis('off')
        
        # Reconstructed images
        axes[1, i].imshow(recon_probs[i].cpu().view(28, 28), cmap='gray')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(f'reconstructions_epoch_{epoch}.png')
    plt.close()

def train_rbm(config):
    train_loader = get_mnist_dataloader(config.batch_size, train=True)
    
    rbm = RBM(config).to(config.device)
    optimizer = torch.optim.Adam(rbm.parameters(), lr=config.learning_rate)
    
    reconstruction_errors = []
    
    for epoch in range(config.num_epochs):
        epoch_error = 0.0
        num_batches = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(config.device)
            data = (data > 0.5).float()  
            
            w_grad, v_bias_grad, h_bias_grad = rbm.contrastive_divergence(data)
            
            optimizer.zero_grad()
            rbm.W.grad = -w_grad
            rbm.v_bias.grad = -v_bias_grad
            rbm.h_bias.grad = -h_bias_grad
            
            optimizer.step()
            
            with torch.no_grad():
                recon_sample, recon_prob, _, _ = rbm.gibbs_sampling(data)
                error = torch.mean((data - recon_sample) ** 2)
                epoch_error += error.item()
                num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{config.num_epochs}] '
                      f'Batch [{batch_idx}/{len(train_loader)}] '
                      f'Error: {error.item():.4f}')
        
        avg_epoch_error = epoch_error / num_batches
        reconstruction_errors.append(avg_epoch_error)
        print(f'Epoch [{epoch+1}/{config.num_epochs}] '
              f'Average Error: {avg_epoch_error:.4f}')
        
        if (epoch + 1) % 5 == 0:
            visualize_reconstructions(rbm, data[:8], epoch+1)
    
    return rbm, reconstruction_errors

def plot_training_curve(reconstruction_errors):
    plt.figure(figsize=(10, 5))
    plt.plot(reconstruction_errors)
    plt.title('RBM Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Error')
    plt.savefig('training_curve.png')
    plt.close()

if __name__ == "__main__":
    
    # Initialize config and train
    config = RBMConfig()
    rbm, reconstruction_errors = train_rbm(config)
    
    # Plot training curve
    plot_training_curve(reconstruction_errors)