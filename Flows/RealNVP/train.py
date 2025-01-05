import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from model import RealNVP
from parser import parse_args

def main():
    args = parse_args()
    
    x, y = make_moons(n_samples=args.n_samples, noise=args.noise)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = RealNVP(input_dim=args.input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        total_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            optimizer.zero_grad()
            log_prob = model.log_prob(x)
            loss = -log_prob.mean()  # Negative log-likelihood
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

    model_path = 'realnvp.pth'
    torch.save(model.state_dict(), model_path)

    train_loader_data = {
        'dataset': dataset,
        'batch_size':train_loader.batch_size,
        'shuffle': train_loader.shuffle
    }

    torch.save(train_loader_data, 'train_loader.pth')
if __name__ == "__main__":
    main()