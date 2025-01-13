import torch
from torch import optim
from config import device, train_loader
from pixel_cnn import PixelCNN
from train import train, plot_training_curve

def train_unconditional_pixelcnn(epochs=100, 
                                lr=0.001, 
                                input_channels=1, 
                                n_hidden=64, 
                                n_layers=8):
    
    model = PixelCNN(
        input_channels=input_channels,
        n_hidden=n_hidden,
        n_layers=n_layers
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training PixelCNN for {epochs} epochs...")
    print(f"Using device: {device}")
    training_losses = train(train_loader, model, optimizer, epochs)
    
    return model, training_losses

if __name__ == "__main__":
    torch.manual_seed(42)
    
    model, losses = train_unconditional_pixelcnn(
        epochs=100,
        lr=0.001,
        input_channels=1,
        n_hidden=64,
        n_layers=8
    )
    
    plot_training_curve(losses)
    
    torch.save(model.state_dict(), 'pixelcnn_model.pth')