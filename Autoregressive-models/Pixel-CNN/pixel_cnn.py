import torch
import torch.nn as nn
from masked_conv import MaskedConv2d

class PixelCNN(nn.Module):
    
    def __init__(self, input_channels=1, n_hidden=64, n_layers=8):
        super().__init__()
        
       
        self.first_conv = MaskedConv2d(
            in_c=input_channels,   
            out_c=n_hidden,        
            kernel_size=7,        
            mask_type='A',         # Type A prevents seeing the current pixel
            padding=3             
        )
        
       
        self.hidden_layers = nn.ModuleList([
            MaskedConv2d(
                in_c=n_hidden,
                out_c=n_hidden,
                kernel_size=3,      
                mask_type='B',      
                padding=1
            ) for _ in range(n_layers)
        ])
        
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(n_hidden) for _ in range(n_layers + 1)
        ])
        
        
        self.output_conv = MaskedConv2d(
            in_c=n_hidden,
            out_c=input_channels,  
            kernel_size=1,        
            mask_type='B'
        )
        
        self.relu = nn.ReLU()
    
    def forward(self, x):

        # Initial feature extraction with type A masking
        x = self.first_conv(x)
        x = self.batch_norms[0](x)
        x = self.relu(x)
        
        for i, (conv, bn) in enumerate(zip(self.hidden_layers, self.batch_norms[1:])):
            residual = x
            
            x = conv(x)
            x = bn(x)
            x = self.relu(x)
            
            # Add residual connection after first layer
            if i > 0:
                x = x + residual
        
        x = self.output_conv(x)
        
       
        return torch.sigmoid(x)