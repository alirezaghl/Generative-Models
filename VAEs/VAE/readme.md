# Variational Autoencoder (VAE) for MNIST

This repository implements a Variational Autoencoder (VAE) in PyTorch for generating and reconstructing MNIST digits. The implementation features two different regularization approaches: Score Function Gradient Estimator (SGVB) and Kullback-Leibler divergence without expectation (KL-WO-E).

## Requirements

```
torch
numpy
matplotlib
torchvision
tqdm
dataclasses
```

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd vae-mnist
```

2. Install the required packages:
```bash
pip install torch torchvision numpy matplotlib tqdm
```


### Configuration

The `Config` dataclass allows you to customize various hyperparameters:

```python
@dataclass
class Config:
    input_dim: int = 784  # 28x28 MNIST images
    hidden_dims = [128, 32, 16, 4]  # Architecture dimensions
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    act_fun: type = nn.GELU
    batch_size: int = 256
    epochs: int = 50
    lr: float = 1e-3
```

### Training

To train the model with both regularization methods:

```python
config = Config()
trainer = Trainer(config)

# Train with SGVB regularization
trainer.train_with_sgvb(MNIST_loader)

# Train with KL divergence regularization
trainer.train_with_kl_wo_e(MNIST_loader)
```

### Visualization

The implementation provides two visualization functions:

1. `evaluate_model()`: Displays original and reconstructed images side by side
2. `plot_latent_images()`: Visualizes the learned latent space by sampling points and generating corresponding images



### Loss Functions

Two regularization approaches are implemented:

1. SGVB (Score Function Gradient Estimator):
   - Implements the score function gradient estimator for the VAE objective
   - Helps maintain a balance between reconstruction quality and latent space structure

2. KL-WO-E (KL divergence without expectation):
   - Implements a simplified version of the KL divergence term
   - Provides an alternative approach to regularizing the latent space

## Resources
1. MIT 6.S978: Deep Generative Models. (Fall 2024). Problem Set 1. *Massachusetts Institute of Technology*.

