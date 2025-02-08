# DDPM MNIST Implementation

A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) trained on the MNIST dataset. This implementation is based on the assignments from 6.S978 Deep Generative Models (MIT EECS, Fall 2024).

## Overview

This implementation includes:
- A U-Net architecture with residual connections and time embeddings
- DDPM training and sampling procedures
- MNIST dataset training pipeline
- Visualization of generated samples and intermediate diffusion steps

## Requirements

- PyTorch
- torchvision
- tqdm
- matplotlib

## Model Architecture

The model consists of:
- Residual convolutional blocks
- U-Net with downsampling and upsampling paths
- Time embedding using fully connected layers
- Skip connections between encoder and decoder

## Usage

1. The model will automatically download the MNIST dataset on first run
2. Training parameters can be modified in the main section:
   - Number of epochs: 20
   - Batch size: 128
   - Learning rate: 1e-4
   - Diffusion steps (T): 400

3. Run the training:
```python
python ddpm_mnist.py
```

## Training Process

- The model saves checkpoints at the final epoch
- Generates sample images every 5 epochs
- Shows intermediate diffusion steps during sampling
- Uses MSE loss for noise prediction
- Implements linear learning rate decay

## References

- 6.S978 Deep Generative Models
- MIT EECS, Fall 2024 pset4