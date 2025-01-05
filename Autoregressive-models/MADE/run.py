import os
import torch
import torch.nn as nn
from model import MADE
from data import get_mnist_loaders
from viz import plot_reconstructions_and_samples, plot_loss_curves

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def train_made_on_mnist(config=None):
    if config is None:
        config = {
            'batch_size': 16,
            'hidden_sizes': [1024, 1024],
            'num_masks': 1,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'num_epochs': 200,
            'mask_change_frequency': 5
        }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get data loaders
    train_loader, test_loader, train_dataset, test_dataset = get_mnist_loaders(config['batch_size'])

    # Create model
    input_size = output_size = 784
    model = MADE(input_size, config['hidden_sizes'], output_size, config['num_masks']).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=config['learning_rate'], 
                                weight_decay=config['weight_decay'])

    train_losses = []
    test_losses = []

    for epoch in range(config['num_epochs']):
        # Training
        model.train()
        total_train_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{config["num_epochs"]}], '
                      f'Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Testing
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                output = model(data)
                loss = criterion(output, data)
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f'Epoch [{epoch+1}/{config["num_epochs"]}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Test Loss: {avg_test_loss:.4f}')

        if (epoch + 1) % config['mask_change_frequency'] == 0:
            model.next_mask()
            print(f"Changed mask at epoch {epoch+1}")
            plot_reconstructions_and_samples(model, test_dataset, device)

    # Plot final loss curves
    plot_loss_curves(train_losses, test_losses)

    return model, train_losses, test_losses

if __name__ == "__main__":
    model, train_losses, test_losses = train_made_on_mnist()
    # Save model
    torch.save(model.state_dict(), 'made_mnist.pth')