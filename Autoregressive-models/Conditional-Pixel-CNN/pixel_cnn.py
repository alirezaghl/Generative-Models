import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

print(torch.cuda.is_available())

class MaskedConvolution(nn.Module):
    def __init__(self, c_in, c_out, mask, **kwargs):
        super().__init__()
        kernel_size = (mask.shape[0], mask.shape[1])
        dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
        padding = tuple([dilation*(kernel_size[i]-1)//2 for i in range(2)])
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, padding=padding, **kwargs)
        self.register_buffer('mask', mask[None,None])

    def forward(self, x):
        self.conv.weight.data *= self.mask
        return self.conv(x)

class VerticalStackConvolution(MaskedConvolution):
    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        mask = torch.ones(kernel_size, kernel_size)
        mask[kernel_size//2+1:,:] = 0
        if mask_center:
            mask[kernel_size//2,:] = 0
        super().__init__(c_in, c_out, mask, **kwargs)

class HorizontalStackConvolution(MaskedConvolution):
    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        mask = torch.ones(1,kernel_size)
        mask[0,kernel_size//2+1:] = 0
        if mask_center:
            mask[0,kernel_size//2] = 0
        super().__init__(c_in, c_out, mask, **kwargs)

class GatedMaskedConv(nn.Module):
    def __init__(self, c_in, num_classes=10, **kwargs):
        super().__init__()
        self.conv_vert = VerticalStackConvolution(c_in, c_out=2*c_in, **kwargs)
        self.conv_horiz = HorizontalStackConvolution(c_in, c_out=2*c_in, **kwargs)
        self.conv_vert_to_horiz = nn.Conv2d(2*c_in, 2*c_in, kernel_size=1, padding=0)
        self.conv_horiz_1x1 = nn.Conv2d(c_in, c_in, kernel_size=1, padding=0)
        
        # Add class conditioning
        self.class_embedding = nn.Embedding(num_classes, c_in)

    def forward(self, v_stack, h_stack, class_labels):
        # Get class embeddings
        class_emb = self.class_embedding(class_labels).unsqueeze(-1).unsqueeze(-1)
        
        # Vertical stack (left)
        v_stack_feat = self.conv_vert(v_stack)
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)
        v_stack_out = torch.tanh(v_val + class_emb) * torch.sigmoid(v_gate)

        # Horizontal stack (right)
        h_stack_feat = self.conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + self.conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        h_stack_feat = torch.tanh(h_val + class_emb) * torch.sigmoid(h_gate)
        h_stack_out = self.conv_horiz_1x1(h_stack_feat)
        h_stack_out = h_stack_out + h_stack

        return v_stack_out, h_stack_out

class ConditionalPixelCNN(pl.LightningModule):
    def __init__(self, c_in, c_hidden):
        super().__init__()
        self.save_hyperparameters()

        # Initial convolutions
        self.conv_vstack = VerticalStackConvolution(c_in, c_hidden, mask_center=True)
        self.conv_hstack = HorizontalStackConvolution(c_in, c_hidden, mask_center=True)
        
        # Gated convolution blocks with dilation
        self.conv_layers = nn.ModuleList([
            GatedMaskedConv(c_hidden),
            GatedMaskedConv(c_hidden, dilation=2),
            GatedMaskedConv(c_hidden),
            GatedMaskedConv(c_hidden, dilation=4),
            GatedMaskedConv(c_hidden),
            GatedMaskedConv(c_hidden, dilation=2),
            GatedMaskedConv(c_hidden)
        ])
        
        self.conv_out = nn.Conv2d(c_hidden, c_in, kernel_size=1, padding=0)

    def forward(self, x, labels):
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack, labels)
        
        # Output layer
        out = self.conv_out(F.elu(h_stack))
        return torch.sigmoid(out)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x, y)
        loss = F.binary_cross_entropy(pred, x)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def generate_samples(self, n_samples=60):
        self.eval()
        device = next(self.parameters()).device
        H, W = 28, 28
        
        samples = torch.zeros(size=(n_samples, 1, H, W)).to(device)
        # Generate 6 samples for each class (0-9)
        labels = torch.tensor(np.repeat(np.arange(10), 6)).to(device)
        
        with torch.no_grad():
            for i in tqdm(range(H)):
                for j in range(W):
                    out = self(samples, labels)
                    samples[:, :, i, j] = torch.bernoulli(out[:, :, i, j])
        
        return samples.cpu().numpy()

    def on_train_epoch_end(self):
        # Generate samples every 2 epochs
        if self.current_epoch % 2 == 1:
            samples = self.generate_samples()
            self._plot_samples(samples)
    
    def _plot_samples(self, samples):
        plt.figure(figsize=(15, 25))
        for i in range(60):
            plt.subplot(10, 6, i + 1)
            plt.imshow(samples[i, 0], cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()

# Data Module
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float())
        ])

    def setup(self, stage=None):
        self.mnist_train = datasets.MNIST("", train=True, download=True, transform=self.transform)
        self.mnist_test = datasets.MNIST("", train=False, download=True, transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_test, batch_size=self.batch_size)


def main():
    # Hyperparameters
    c_in = 1 
    c_hidden = 64  
    batch_size = 128
    max_epochs = 20

    # Create model and data module
    model = ConditionalPixelCNN(c_in=c_in, c_hidden=c_hidden)
    data_module = MNISTDataModule(batch_size=batch_size)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto', 
        devices=1,
        enable_progress_bar=True
    )

    # Train model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()