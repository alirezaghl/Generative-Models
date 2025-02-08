# DDPM MNIST Implementation

A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) trained on the MNIST dataset. This implementation is based on the assignments from 6.S978 Deep Generative Models (MIT EECS, Fall 2024).

## Overview

This implementation includes:
- A U-Net architecture with residual connections and time embeddings
- DDPM training and sampling procedures
- Visualization of generated samples and intermediate diffusion steps

## Limitations and Future Improvements

This implementation is a basic example and not a state-of-the-art model. Areas for improvement include:

- Limited network capacity (using basic U-Net architecture)
- No advanced sampling techniques like DDIM
- Basic time embeddings without attention mechanisms
- No conditioning capabilities
- Limited hyperparameter tuning
- Basic learning rate schedule

  
## References

- 6.S978 Deep Generative Models
- MIT EECS, Fall 2024 pset4
