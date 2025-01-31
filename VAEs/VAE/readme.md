# Variational Autoencoder (VAE) for MNIST

This repository implements a Variational Autoencoder (VAE) in PyTorch for generating and reconstructing MNIST digits. The implementation features two different regularization approaches: Score Function Gradient Estimator (SGVB) and Kullback-Leibler divergence without expectation (KL-WO-E).

## Overview

Variational Autoencoders are powerful generative models that learn to compress data into a lower-dimensional latent space while maintaining the ability to reconstruct the original input. This implementation includes:

- A configurable VAE architecture with customizable hidden dimensions
- Two regularization methods: SGVB and KL-WO-E
- Visualization tools for both original and reconstructed images
- Latent space visualization capabilities
- Training progress monitoring with tqdm

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

## Usage

The main components of the implementation are:

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

## Architecture Details

### VAE Structure

The VAE consists of:

- Encoder: Compresses input images into a latent representation
- Reparameterization layer: Enables backpropagation through random sampling
- Decoder: Reconstructs images from the latent representation

### Loss Functions

Two regularization approaches are implemented:

1. SGVB (Score Function Gradient Estimator):
   - Implements the score function gradient estimator for the VAE objective
   - Helps maintain a balance between reconstruction quality and latent space structure

2. KL-WO-E (KL divergence without expectation):
   - Implements a simplified version of the KL divergence term
   - Provides an alternative approach to regularizing the latent space

## Results

During training, you will see:

- Training progress with loss values for each epoch
- Visualization of original vs reconstructed images
- Latent space visualization showing the learned manifold of digit representations

## Customization

You can customize the model by:

1. Modifying the hidden dimensions in the Config class
2. Changing the activation function (default is GELU)
3. Adjusting the learning rate and batch size
4. Modifying the number of training epochs

## Contributing

Feel free to submit issues and enhancement requests! To contribute:

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

[Add your chosen license here]

## Acknowledgments

This implementation draws inspiration from the original VAE paper:
"Auto-Encoding Variational Bayes" by Kingma and Welling.
