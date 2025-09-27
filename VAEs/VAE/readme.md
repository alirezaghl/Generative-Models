# Variational Autoencoder (VAE) for MNIST

this repository implements a Variational Autoencoder (VAE) in PyTorch for generating and reconstructing MNIST digits. The implementation features two different regularization approaches: Score Function Gradient Estimator (SGVB) and Kullback-Leibler divergence without expectation (KL-WO-E).


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

