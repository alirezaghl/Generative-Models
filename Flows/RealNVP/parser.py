import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="RealNVP model implementation")
    parser.add_argument('--n_samples', type=int, default=3000, help='Number of samples to generate')
    parser.add_argument('--noise', type=float, default=0.05, help='Noise level for data generation')
    parser.add_argument('--input_dim', type=int, default=2, help='Input dimensionality')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimensionality')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of coupling layers')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    return parser.parse_args()