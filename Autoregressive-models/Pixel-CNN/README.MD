# PixelCNN Implementation

This repository contains an implementation of PixelCNN using PyTorch Lightning.

## Overview

Conditional PixelCNN is an autoregressive model that generates images pixel by pixel

![Sample Results](https://github.com/alirezaghl/Generative-Models/blob/main/Autoregressive-models/Pixel-CNN/results-gated-pixelcnn/samples_epcoh_20.png)

## Key Features

- **Masked Convolutions**: Implementation of both vertical and horizontal masked convolutions to ensure proper autoregressive generation
- **Gated Activation Units**: Using gated convolutions with tanh and sigmoid activations
- **Class Conditioning**: Embedding layer for class conditioning integrated into the generation process
- **Dilated Convolutions**: Multiple layers with different dilation rates to increase receptive field
- **PyTorch Lightning Integration**: Clean, organized code structure with built-in training utilities

## References

[1] Van den Oord, A., Kalchbrenner, N., & Kavukcuoglu, K. (2016). "Pixel Recurrent Neural Networks." *International Conference on Machine Learning (ICML)*. [[paper]](https://arxiv.org/abs/1601.06759)

[2] Van den Oord, A., Kalchbrenner, N., Espeholt, L., Vinyals, O., & Graves, A. (2016). "Conditional Image Generation with PixelCNN Decoders." *Advances in Neural Information Processing Systems (NeurIPS)*. [[paper]](https://arxiv.org/abs/1606.05328)

[3] Phillip Lippe. (2022). "Tutorial 12: Autoregressive Image Modeling." *UvA Deep Learning Course*. [[tutorial]](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial12/Autoregressive_Image_Modeling.html)

[4] MIT 6.S978: Deep Generative Models. (Fall 2024). Problem Set 2. *Massachusetts Institute of Technology*.
